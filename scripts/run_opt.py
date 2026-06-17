"""Multi-seed runner for the LEO optimization-surrogate study.

These models are tiny (hundreds of params) so the whole sweep runs in ~minutes on
CPU; GPU is optional. Writes one CSV row per (method, seed) + per-slot logs.

  python scripts/run_opt.py --seeds 5 --out results/opt.csv
"""
import sys, os, csv, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from evokan.methods import make_config, METHODS
from evokan.experiment_opt import run_experiment
from evokan.symbolic import extract_rules


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-nodes", type=int, default=10)
    p.add_argument("--slots-per-pass", type=int, default=24)
    p.add_argument("--n-passes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--slot-samples", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--grid-max", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--csi-sigma", type=float, default=0.0)
    p.add_argument("--heterogeneous", action="store_true",
                   help="per-node g/p0 sub-ranges (non-IID); default homogeneous")
    p.add_argument("--append", action="store_true",
                   help="append to an existing --out CSV instead of overwriting")
    p.add_argument("--log-every", type=int, default=0,
                   help="print an in-run heartbeat every N logged slots (0=off)")
    p.add_argument("--out", default="results/opt.csv")
    p.add_argument("--logdir", default="results/logs")
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.logdir, exist_ok=True)
    rows = []
    if a.append and os.path.exists(a.out):          # keep existing rows, append new runs
        with open(a.out) as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
        keep_methods = set(a.methods)
        rows = [r for r in rows if r.get("method") not in keep_methods]  # drop stale dups
        print(f"append mode: {len(rows)} existing rows kept")
    common = dict(device=a.device, n_nodes=a.n_nodes, slots_per_pass=a.slots_per_pass,
                  n_passes=a.n_passes, epochs=a.epochs, slot_samples=a.slot_samples,
                  grid_size=a.grid_size, grid_max=a.grid_max, lr=a.lr,
                  csi_sigma=a.csi_sigma, heterogeneous=a.heterogeneous)

    def flush():
        if rows:
            with open(a.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    total = len(a.methods) * a.seeds
    done = 0
    for method in a.methods:
        for seed in range(a.seeds):
            done += 1
            cfg = make_config(method, seed=seed, **common)
            print(f"[{done}/{total}] running {method} seed={seed} ...", flush=True)
            t0 = time.time()
            try:
                res, log, model = run_experiment(cfg, log_every=a.log_every,
                                                 tag=f"{method}/s{seed}")
            except Exception as e:
                import traceback; print(f"!! {method} s{seed} FAILED: {e}", flush=True)
                traceback.print_exc(); continue
            rule_r2 = ""
            if cfg.model_type == "kan":
                _, rule_r2 = extract_rules(model)
                rule_r2 = round(rule_r2, 4)
            row = {"method": method, "seed": seed, "time_s": round(time.time() - t0, 1),
                   "rule_r2": rule_r2, **{k: res[k] for k in
                   ["avg_gap", "final_gap", "avg_nmse", "final_nmse", "forgetting",
                    "detection_delay", "total_bits", "final_grid", "params"]}}
            rows.append(row); flush()
            np.savez(os.path.join(a.logdir, f"{method}_s{seed}.npz"),
                     **{k: np.array(v, dtype=object) for k, v in log.items()})
            print(f"[{done}/{total}] {method:12s} s{seed} nmse={res['final_nmse']:.4f} "
                  f"gap={res['avg_gap']:.4f} bits={res['total_bits']:.2e} "
                  f"ruleR2={rule_r2} ({row['time_s']}s)", flush=True)

    # summary
    import statistics as st
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ["final_nmse", "avg_gap", "total_bits"]:
            agg[r["method"]][k].append(float(r[k]))
    print("\n=== summary (mean +/- std) ===")
    for m, d in agg.items():
        nm = d["final_nmse"]
        print(f"{m:12s} nmse={st.mean(nm):.4f}+/-{st.pstdev(nm):.4f} "
              f"gap={st.mean(d['avg_gap']):.4f} bits={st.mean(d['total_bits']):.2e}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
