"""Local CPU smoke test for the optimization-surrogate pipeline.

Runs ours + key baselines on a tiny config, prints regret/NMSE + KAN rule
fidelity, and sanity-checks wiring. Not a result.

Run:  python scripts/smoke_opt.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evokan.methods import make_config
from evokan.experiment_opt import run_experiment
from evokan.symbolic import extract_rules

TINY = dict(n_nodes=4, slots_per_pass=10, n_passes=2, gap_slots=2,
            slot_samples=64, epochs=3, grid_size=4, grid_max=8, hidden=(8,))


def main():
    print("=== EvoKAN smoke test (LEO EE power-opt surrogate, CPU) ===")
    ok = True
    for method in ["fedkan_opt", "kan_full", "mlp"]:
        cfg = make_config(method, **TINY, seed=0)
        t0 = time.time()
        res, log, model = run_experiment(cfg)
        dt = time.time() - t0
        print(f"\n[{method}]  ({dt:.1f}s)")
        for k in ["avg_gap", "final_gap", "final_nmse", "forgetting", "total_bits",
                  "final_grid", "params"]:
            print(f"   {k:12s} = {res[k]}")
        if cfg.model_type == "kan":
            rules, fid = extract_rules(model)
            print(f"   rule_fidelity (mean R2) = {fid:.3f}")
        if res["total_bits"] <= 0:
            print("   FAIL: no uplink bits"); ok = False
        if res["final_gap"] > 1.0 or res["final_gap"] < 0:
            print("   FAIL: gap out of range"); ok = False
    print("\nSMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
