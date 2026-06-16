"""Depth studies for the JSTSP expansion, all on the single-link EE problem where
the KAN surrogate wins. Each is multi-seed and returns raw per-seed numbers.

  study_generalization  - train on a subset of interference regimes, test on an
                          UNSEEN regime (extrapolation). KAN splines extrapolate
                          far better than an MLP.
  study_csi_robustness  - deployment-time CSI estimation error on the gain input;
                          does the KAN advantage survive noisy state?
  study_pareto          - federated comm-vs-accuracy frontier by sweeping the
                          update keep-fraction.
  study_scaling         - final accuracy / uplink vs number of terminals.
  study_capacity        - centralized NMSE vs KAN grid size (supports "a static
                          grid suffices, self-evolution is unnecessary").
"""
from __future__ import annotations
import numpy as np
import torch
from . import opt_problem as op
from .kan import KANClassifier, mlp_matching_kan
from . import fed
from .methods import make_config
from .experiment_opt import run_experiment


def _central_pair(seed, grid_size=6):
    torch.manual_seed(seed)
    kan = KANClassifier(3, (10,), 1, grid_size=grid_size)
    mlp, _ = mlp_matching_kan(3, 1, kan)
    return kan, mlp


def _draw(rng, levels, n, st):
    g = np.exp(rng.uniform(np.log(0.2), np.log(30), n))
    p0 = rng.uniform(0.1, 1.0, n)
    lv = np.asarray(levels, float)
    I = rng.choice(lv, n) if lv.ndim and lv.size > 1 else np.full(n, float(lv))
    y, _ = op.optimal_power(g, I, p0)
    X = op.standardize_states(np.stack([g, I, p0], 1).astype("float32"), st)
    return X, y.astype("float32"), g, I, p0


def study_generalization(seeds=10, levels_train=(0.2, 1.0), level_test=3.0):
    st = op.feature_stats()
    out = {"kan": {"indom": [], "extrap": []}, "mlp": {"indom": [], "extrap": []}}
    for s in range(seeds):
        print(f"   [generalization] seed {s+1}/{seeds}", flush=True)
        rng = np.random.default_rng(s)
        Xtr, ytr, *_ = _draw(rng, levels_train, 4000, st)
        Xin, yin, *_ = _draw(rng, levels_train, 1500, st)
        Xex, yex, *_ = _draw(rng, level_test, 1500, st)
        kan, mlp = _central_pair(s)
        fed.local_train(kan, Xtr, ytr, epochs=80, lr=0.01, task="reg")
        fed.local_train(mlp, Xtr, ytr, epochs=80, lr=0.01, task="reg")
        out["kan"]["indom"].append(fed.nmse(kan, Xin, yin))
        out["kan"]["extrap"].append(fed.nmse(kan, Xex, yex))
        out["mlp"]["indom"].append(fed.nmse(mlp, Xin, yin))
        out["mlp"]["extrap"].append(fed.nmse(mlp, Xex, yex))
    return out


def study_csi_robustness(seeds=10, noise_db=(0, 1, 2, 4, 6)):
    st = op.feature_stats()
    res = {nd: {"kan": [], "mlp": []} for nd in noise_db}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        Xtr, ytr, *_ = _draw(rng, (0.2, 1.0, 3.0), 4000, st)
        kan, mlp = _central_pair(s)
        fed.local_train(kan, Xtr, ytr, epochs=80, lr=0.01, task="reg")
        fed.local_train(mlp, Xtr, ytr, epochs=80, lr=0.01, task="reg")
        gt = np.exp(rng.uniform(np.log(0.2), np.log(30), 1500))
        It = rng.uniform(0.1, 3, 1500); p0t = rng.uniform(0.1, 1, 1500)
        yt, _ = op.optimal_power(gt, It, p0t); yt = yt.astype("float32")
        for nd in noise_db:
            fade = 10.0 ** (nd * rng.standard_normal(1500) / 10.0)   # CSI error on gain
            gh = np.clip(gt * fade, 1e-3, None)
            Xt = op.standardize_states(np.stack([gh, It, p0t], 1).astype("float32"), st)
            res[nd]["kan"].append(fed.nmse(kan, Xt, yt))
            res[nd]["mlp"].append(fed.nmse(mlp, Xt, yt))
    return res


def study_pareto(seeds=5, keeps=(0.1, 0.25, 0.5, 0.75, 1.0), **kw):
    pts = {k: {"bits": [], "nmse": []} for k in keeps}
    bandit = {"bits": [], "nmse": []}
    for s in range(seeds):
        print(f"   [pareto] seed {s+1}/{seeds}", flush=True)
        for k in keeps:
            cfg = make_config("kan_full", seed=s, fixed_keep=k, **kw)
            res, _, _ = run_experiment(cfg)
            pts[k]["bits"].append(res["total_bits"]); pts[k]["nmse"].append(res["final_nmse"])
        cfg = make_config("fedkan_opt", seed=s, **kw)
        res, _, _ = run_experiment(cfg)
        bandit["bits"].append(res["total_bits"]); bandit["nmse"].append(res["final_nmse"])
    return pts, bandit


def study_per_regime(seeds=5, methods=("fedkan_opt", "kan_full", "mlp"),
                     levels=(0.2, 1.0, 3.0), **kw):
    """Final NMSE per interference regime for each method (multi-seed)."""
    out = {m: {L: [] for L in levels} for m in methods}
    for m in methods:
        print(f"   [per-regime] {m}", flush=True)
        for s in range(seeds):
            cfg = make_config(m, seed=s, interference_levels=tuple(levels), **kw)
            res, _, _ = run_experiment(cfg)
            for L in levels:
                v = res["regime_nmse"].get(float(L))
                if v is not None and v == v:   # not NaN
                    out[m][L].append(v)
    return out


def study_csi_fed(seeds=5, sigmas=(0.0, 0.5, 1.0, 2.0),
                  methods=("fedkan_opt", "mlp"), **kw):
    """CSI estimation-error sensitivity in the FEDERATED loop: surrogate sees a
    noisy gain (csi_sigma>0), oracle labels use the true gain."""
    out = {m: {sg: [] for sg in sigmas} for m in methods}
    for sg in sigmas:
        print(f"   [csi-fed] sigma={sg}", flush=True)
        for m in methods:
            for s in range(seeds):
                cfg = make_config(m, seed=s, csi_sigma=sg, **kw)
                res, _, _ = run_experiment(cfg)
                out[m][sg].append(res["final_nmse"])
    return out


def study_scaling(seeds=5, nodes=(4, 8, 12, 16, 20), **kw):
    out = {n: {"nmse": [], "bits": []} for n in nodes}
    for n in nodes:
        print(f"   [scaling] n_nodes={n}", flush=True)
        for s in range(seeds):
            cfg = make_config("fedkan_opt", seed=s, n_nodes=n, **kw)
            res, _, _ = run_experiment(cfg)
            out[n]["nmse"].append(res["final_nmse"]); out[n]["bits"].append(res["total_bits"])
    return out


def study_capacity(seeds=10, grids=(3, 5, 7, 9, 12)):
    st = op.feature_stats()
    out = {g: {"nmse": [], "params": 0} for g in grids}
    for gs in grids:
        print(f"   [capacity] grid={gs}", flush=True)
        for s in range(seeds):
            rng = np.random.default_rng(s)
            Xtr, ytr, *_ = _draw(rng, (0.2, 1.0, 3.0), 4000, st)
            Xte, yte, *_ = _draw(rng, (0.2, 1.0, 3.0), 1500, st)
            torch.manual_seed(s)
            m = KANClassifier(3, (10,), 1, grid_size=gs)
            fed.local_train(m, Xtr, ytr, epochs=80, lr=0.01, task="reg")
            out[gs]["nmse"].append(fed.nmse(m, Xte, yte)); out[gs]["params"] = m.num_params()
    return out
