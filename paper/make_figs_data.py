"""Generate all figure data (.dat) for the manuscript from results + seeded runs.

Outputs into paper/data/:
  summary.dat   per-method mean/std of nmse, regret, bits, ruleR2 (from results CSV)
  ts_<m>.dat    per-slot NMSE/gap time series for a representative seed
  rule_<f>.dat  KAN univariate response curves (g, I, p0) + best-fit form
  rules.txt     extracted closed-form summary
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, statistics as st
from collections import defaultdict
from evokan.methods import make_config
from evokan.experiment_opt import run_experiment
from evokan.symbolic import extract_rules, _fit_form, _LIB

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
CSV = os.path.join(HERE, "..", "results", "opt.csv")
M = ["fedkan_opt", "kan_full", "kan_evolve", "mlp", "mlp_prox", "linear"]


def summary():
    rows = list(csv.DictReader(open(CSV)))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ["final_nmse", "avg_gap", "total_bits", "rule_r2", "params"]:
            if r[k] not in ("", "nan"):
                agg[r["method"]][k].append(float(r[k]))
    with open(os.path.join(DATA, "summary.dat"), "w") as f:
        f.write("method nmse nmse_sd regret bits ruleR2 params\n")
        for m in M:
            d = agg[m]
            rr = st.mean(d["rule_r2"]) if d["rule_r2"] else -1
            f.write("%s %.5f %.5f %.5f %.4e %.3f %d\n" % (
                m, st.mean(d["final_nmse"]), st.pstdev(d["final_nmse"]),
                st.mean(d["avg_gap"]), st.mean(d["total_bits"]), rr,
                int(st.mean(d["params"]))))
    print("wrote summary.dat")


def timeseries():
    kw = dict(n_nodes=10, slots_per_pass=24, n_passes=4, epochs=6, slot_samples=128,
              grid_size=5, grid_max=10, seed=3, device="cpu")
    model_full = None
    for m in ["fedkan_opt", "kan_full", "mlp"]:
        res, log, model = run_experiment(make_config(m, **kw))
        if m == "kan_full":
            model_full = model
        with open(os.path.join(DATA, "ts_%s.dat" % m), "w") as f:
            f.write("slot nmse gap\n")
            for i, s in enumerate(log["slot"]):
                if log["nmse"][i] is None:
                    continue
                f.write("%d %.5f %.5f\n" % (s, log["nmse"][i], log["gap"][i]))
        print("wrote ts_%s.dat" % m)
    return model_full


@torch.no_grad()
def rule_curves(model):
    layer = model.layers[0]
    dev = next(model.parameters()).device
    names = ["g", "I", "p0"]
    xs = torch.linspace(-2, 2, 200, device=dev)
    _, fid = extract_rules(model)
    lines = ["mean R2=%.3f" % fid]
    for i in range(3):
        X = torch.zeros(200, layer.in_features, device=dev); X[:, i] = xs
        y = layer(X).sum(1).cpu().numpy()
        xn = xs.cpu().numpy()
        form, r2, a, b = _fit_form(xn, y)
        fit = a * _LIB[form](xn) + b
        with open(os.path.join(DATA, "rule_%s.dat" % names[i]), "w") as f:
            f.write("x y yfit\n")
            for k in range(len(xn)):
                f.write("%.4f %.5f %.5f\n" % (xn[k], y[k], fit[k]))
        lines.append("%s: %s (R2=%.3f)" % (names[i], form, r2))
        print("wrote rule_%s.dat  form=%s R2=%.3f" % (names[i], form, r2))
    open(os.path.join(DATA, "rules.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    summary()
    model = timeseries()
    rule_curves(model)
    print("DONE")
