"""Federated online learning of the optimization solution map over an LEO pass.

Each slot, visible terminals draw fresh link states at their current geometry +
interference regime, label them with the oracle optimal power, and update a shared
KAN (or MLP) surrogate. A drift detector on the surrogate error triggers
self-evolution (grid growth) so the compact surrogate tracks the drifting optimal
policy; a bandit controller trades grid size / update sparsity against an uplink
budget; updates are FedAvg-aggregated across heterogeneous geometries.

Headline metric = optimality gap to the oracle (utility regret) and NMSE on p*,
where the KAN wins; plus retention across recurring interference regimes
(forgetting), communication bits, and drift-detection delay.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from . import opt_problem as op
from . import orbit
from .kan import KANClassifier, MLPClassifier, mlp_matching_kan
from .drift import make_detector
from .controller import make_controller
from . import fed


@dataclass
class Config:
    # topology / orbit
    n_nodes: int = 10
    slots_per_pass: int = 30
    n_passes: int = 4
    gap_slots: int = 4
    theta_min: float = 10.0
    snr_zenith: float = 22.0
    # optimization
    Pmax: float = 5.0
    interference_levels: tuple = (0.2, 1.0, 3.0)
    min_dwell: int = 8
    # model
    model_type: str = "kan"          # kan | mlp
    hidden: tuple = (10,)
    grid_size: int = 5
    grid_max: int = 12
    spline_order: int = 3
    # learning / FL
    epochs: int = 5
    lr: float = 0.01
    mu: float = 0.0
    l1: float = 0.0
    weight_decay: float = 1e-4
    slot_samples: int = 96
    max_eval: int = 400
    # control
    controller: str = "bandit"
    fixed_keep: float = 1.0          # keep-fraction for FixedController (Pareto sweep)
    detector: str = "page_hinkley"
    evolve_enabled: bool = True
    budget_bits: float = 1.0e5
    comm_lambda: float = 0.3
    bits_per_param: int = 32
    # misc
    seed: int = 0
    device: str = "cpu"


def _regressor(cfg, in_dim):
    if cfg.model_type == "kan":
        return KANClassifier(in_dim, cfg.hidden, 1, grid_size=cfg.grid_size,
                             spline_order=cfg.spline_order)
    if cfg.model_type == "linear":
        return MLPClassifier(in_dim, [], 1)        # single Linear layer = linear reg
    ref = KANClassifier(in_dim, cfg.hidden, 1, grid_size=cfg.grid_size)
    m, _ = mlp_matching_kan(in_dim, 1, ref)
    return m


def _eval_battery(cfg, rng):
    """Held-out states per interference regime (for regret + forgetting)."""
    stats = op.feature_stats(cfg.Pmax)
    by_regime = {}
    for L in cfg.interference_levels:
        g = rng.uniform(0.2, 30.0, cfg.max_eval)
        p0 = rng.uniform(0.1, 1.0, cfg.max_eval)
        I = np.full(cfg.max_eval, L)
        pstar, eestar = op.optimal_power(g, I, p0, Pmax=cfg.Pmax)
        X = np.stack([g, I, p0], 1).astype(np.float32)
        by_regime[L] = dict(X=X, Xs=op.standardize_states(X, stats),
                            pstar=pstar.astype(np.float32), eestar=eestar.astype(np.float32),
                            g=g, I=I, p0=p0)
    return by_regime, stats


def _utility_gap(model, reg, cfg, device):
    """Mean relative EE regret = (EE(p*) - EE(p_pred)) / EE(p*) on a regime set."""
    p_pred = np.clip(fed.predict_reg(model, reg["Xs"], device), 1e-3, cfg.Pmax)
    ee_pred = op.ee_utility(p_pred, reg["g"], reg["I"], reg["p0"])
    gap = (reg["eestar"] - ee_pred) / (reg["eestar"] + 1e-12)
    return float(np.mean(np.clip(gap, 0, None)))


def run_experiment(cfg: Config, verbose=False):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    stats = op.feature_stats(cfg.Pmax)

    # nodes: geometry + per-node interference schedule
    T_pass = cfg.slots_per_pass
    nodes = []
    for _ in range(cfg.n_nodes):
        g_series, el, vis = op.pass_gain(T_pass, rng.uniform(25, 85), cfg.snr_zenith, cfg.theta_min)
        # tile passes with gaps
        g_full, vis_full = [], []
        for _p in range(cfg.n_passes):
            g_full.append(g_series); vis_full.append(vis)
            if cfg.gap_slots:
                g_full.append(np.zeros(cfg.gap_slots)); vis_full.append(np.zeros(cfg.gap_slots, bool))
        g_full = np.concatenate(g_full); vis_full = np.concatenate(vis_full)
        I_sched, switches = op.interference_schedule(len(g_full), rng,
                                                     cfg.interference_levels, cfg.min_dwell)
        nodes.append(dict(g=g_full, vis=vis_full, I=I_sched, p0=float(rng.uniform(0.1, 1.0)),
                          switches=switches))
    T = len(nodes[0]["g"])

    eval_by_regime, _ = _eval_battery(cfg, np.random.default_rng(cfg.seed + 3))
    drift_truth = [False] * T
    for nd in nodes:
        for t in nd["switches"]:
            if t < T:
                drift_truth[t] = True

    global_model = _regressor(cfg, 3)
    detectors = [make_detector(cfg.detector) for _ in range(cfg.n_nodes)]
    controllers = [make_controller(cfg.controller, keep=cfg.fixed_keep)
                   for _ in range(cfg.n_nodes)]

    log = {"slot": [], "gap": [], "nmse": [], "bits": [], "grid": [],
           "n_part": [], "drift_flags": []}
    regime_nmse_log = {L: [] for L in cfg.interference_levels}
    budget_price = 0.0

    def cumulative_gap():
        return float(np.mean([_utility_gap(global_model, eval_by_regime[L], cfg, cfg.device)
                              for L in cfg.interference_levels]))

    def cumulative_nmse():
        return float(np.mean([fed.nmse(global_model, eval_by_regime[L]["Xs"],
                                       eval_by_regime[L]["pstar"], cfg.device)
                              for L in cfg.interference_levels]))

    for t in range(T):
        participants, slot_drift = [], 0
        for nidx in range(cfg.n_nodes):
            nd = nodes[nidx]
            if not nd["vis"][t]:
                continue
            X, y, _ = op.sample_states(nd["I"][t], cfg.slot_samples, rng, Pmax=cfg.Pmax)
            Xs = op.standardize_states(X, stats)
            local = fed.clone_model(global_model)
            pre_nmse = fed.nmse(local, Xs, y, cfg.device)
            flag = detectors[nidx].update(pre_nmse)
            slot_drift = max(slot_drift, int(flag))

            ctx = {"drift": float(flag),
                   "snr_norm": float(np.clip(nd["g"][t] / 30.0, 0, 1)),
                   "grid_norm": (getattr(local, "grid_size", cfg.grid_size) - cfg.grid_size)
                                / max(1, cfg.grid_max - cfg.grid_size),
                   "budget_price": budget_price,
                   "acc": float(np.exp(-pre_nmse))}
            evolve_step, keep = controllers[nidx].act(ctx)
            if cfg.evolve_enabled and cfg.model_type == "kan" and evolve_step > 0 and flag:
                fed.align_to_grid(local, min(cfg.grid_max, local.grid_size + evolve_step))

            gvec = None
            if cfg.mu > 0:
                gc = fed.clone_model(global_model)
                if hasattr(local, "evolve"):
                    fed.align_to_grid(gc, local.grid_size)
                gvec = parameters_to_vector(gc.parameters()).detach()
            fed.local_train(local, Xs, y, cfg.epochs, cfg.lr, cfg.mu, gvec,
                            cfg.l1, cfg.weight_decay, device=cfg.device, task="reg")
            participants.append({"model": local, "weight": cfg.slot_samples, "keep": keep,
                                 "nidx": nidx, "ctx": ctx, "action": (evolve_step, keep),
                                 "pre": pre_nmse})

        if not participants:
            for k, v in [("slot", t), ("gap", None), ("nmse", None), ("bits", 0),
                         ("grid", getattr(global_model, "grid_size", 0)),
                         ("n_part", 0), ("drift_flags", 0)]:
                log[k].append(v)
            continue

        gap_before = cumulative_gap()
        total_bits, _ = fed.fed_aggregate(global_model, participants, cfg.bits_per_param, cfg.mu)
        gap_after = cumulative_gap()
        reward = (gap_before - gap_after) - cfg.comm_lambda * (total_bits / cfg.budget_bits)
        for p in participants:
            controllers[p["nidx"]].update(p["ctx"], p["action"], reward)
        budget_price = float(np.clip(total_bits / cfg.budget_bits, 0, 1))

        for L in cfg.interference_levels:
            regime_nmse_log[L].append((t, fed.nmse(global_model, eval_by_regime[L]["Xs"],
                                                   eval_by_regime[L]["pstar"], cfg.device)))
        log["slot"].append(t); log["gap"].append(gap_after); log["nmse"].append(cumulative_nmse())
        log["bits"].append(total_bits); log["grid"].append(getattr(global_model, "grid_size", 0))
        log["n_part"].append(len(participants)); log["drift_flags"].append(slot_drift)
        if verbose:
            print(f"t={t:3d} part={len(participants)} gap={gap_after:.4f} "
                  f"nmse={log['nmse'][-1]:.4f} grid={log['grid'][-1]} bits={total_bits}")

    # forgetting on recurring regimes: mean (max - final) NMSE
    forget = []
    for L, seq in regime_nmse_log.items():
        if len(seq) >= 2:
            v = [a for _, a in seq]; forget.append(max(v) - v[-1])
    from .drift import PageHinkley  # noqa (kept for parity)
    flags_series = [0] * T
    for s, f in zip(log["slot"], log["drift_flags"]):
        flags_series[s] = int(f)
    delays = []
    fl = np.where(np.asarray(flags_series))[0]
    for td in np.where(np.asarray(drift_truth))[0]:
        later = fl[fl >= td]
        if len(later):
            delays.append(int(later[0] - td))

    gaps = [g for g in log["gap"] if g is not None]
    nmses = [m for m in log["nmse"] if m is not None]
    result = {
        "avg_gap": float(np.mean(gaps)) if gaps else 1.0,
        "final_gap": cumulative_gap(),
        "avg_nmse": float(np.mean(nmses)) if nmses else 1.0,
        "final_nmse": cumulative_nmse(),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "total_bits": float(np.sum(log["bits"])),
        "final_grid": log["grid"][-1] if log["grid"] else 0,
        "params": global_model.num_params(), "T": T,
    }
    return result, log, global_model
