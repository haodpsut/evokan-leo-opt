"""Run the depth studies and write figure data into paper/data/.

  python scripts/run_studies.py            # default seeds
  python scripts/run_studies.py --quick     # fewer seeds for a fast check
"""
import sys, os, argparse, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from evokan import studies

DATA = os.path.join(os.path.dirname(__file__), "..", "paper", "data")
os.makedirs(DATA, exist_ok=True)


def ms(xs):
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    S = 3 if a.quick else 10
    Sf = 3 if a.quick else 5
    fed_kw = dict(n_nodes=10, slots_per_pass=24, n_passes=4, epochs=6, slot_samples=128)

    # A. generalization
    g = studies.study_generalization(seeds=S)
    with open(os.path.join(DATA, "gen.dat"), "w") as f:
        f.write("method indom indom_sd extrap extrap_sd\n")
        for m in ["kan", "mlp"]:
            im, isd = ms(g[m]["indom"]); em, esd = ms(g[m]["extrap"])
            f.write("%s %.5f %.5f %.5f %.5f\n" % (m, im, isd, em, esd))
    print("A generalization:", {m: (round(ms(g[m]['indom'])[0], 3), round(ms(g[m]['extrap'])[0], 3)) for m in g})

    # B. CSI robustness
    nd = (0, 1, 2, 4, 6)
    r = studies.study_csi_robustness(seeds=S, noise_db=nd)
    with open(os.path.join(DATA, "csi.dat"), "w") as f:
        f.write("noise_db kan kan_sd mlp mlp_sd\n")
        for n in nd:
            km, ksd = ms(r[n]["kan"]); mm, msd = ms(r[n]["mlp"])
            f.write("%d %.5f %.5f %.5f %.5f\n" % (n, km, ksd, mm, msd))
    print("B csi done")

    # C. Pareto
    keeps = (0.1, 0.25, 0.5, 0.75, 1.0)
    pts, bandit = studies.study_pareto(seeds=Sf, keeps=keeps, **fed_kw)
    with open(os.path.join(DATA, "pareto.dat"), "w") as f:
        f.write("keep bits nmse\n")
        for k in keeps:
            f.write("%.2f %.4e %.5f\n" % (k, ms(pts[k]["bits"])[0], ms(pts[k]["nmse"])[0]))
    with open(os.path.join(DATA, "pareto_bandit.dat"), "w") as f:
        f.write("bits nmse\n%.4e %.5f\n" % (ms(bandit["bits"])[0], ms(bandit["nmse"])[0]))
    print("C pareto done")

    # D. scaling
    nodes = (4, 8, 12, 16, 20)
    sc = studies.study_scaling(seeds=Sf, nodes=nodes, slots_per_pass=24, n_passes=4, epochs=6)
    with open(os.path.join(DATA, "scaling.dat"), "w") as f:
        f.write("nodes nmse nmse_sd bits\n")
        for n in nodes:
            nm, nsd = ms(sc[n]["nmse"])
            f.write("%d %.5f %.5f %.4e\n" % (n, nm, nsd, ms(sc[n]["bits"])[0]))
    print("D scaling done")

    # E. capacity
    grids = (3, 5, 7, 9, 12)
    cap = studies.study_capacity(seeds=S, grids=grids)
    with open(os.path.join(DATA, "capacity.dat"), "w") as f:
        f.write("grid nmse nmse_sd params\n")
        for gs in grids:
            nm, nsd = ms(cap[gs]["nmse"])
            f.write("%d %.5f %.5f %d\n" % (gs, nm, nsd, cap[gs]["params"]))
    print("E capacity:", {gs: round(ms(cap[gs]['nmse'])[0], 3) for gs in grids})

    # F. per-regime breakdown
    levels = (0.2, 1.0, 3.0)
    pr = studies.study_per_regime(seeds=Sf, levels=levels, **fed_kw)
    with open(os.path.join(DATA, "regime.dat"), "w") as f:
        f.write("regime fedkan_opt kan_full mlp\n")
        for L in levels:
            f.write("%.1f %.5f %.5f %.5f\n" % (L, ms(pr["fedkan_opt"][L])[0],
                    ms(pr["kan_full"][L])[0], ms(pr["mlp"][L])[0]))
    print("F per-regime done")

    # G. CSI-error sensitivity (federated)
    sigmas = (0.0, 0.5, 1.0, 2.0)
    cf = studies.study_csi_fed(seeds=Sf, sigmas=sigmas, **fed_kw)
    with open(os.path.join(DATA, "csi_fed.dat"), "w") as f:
        f.write("sigma kan kan_sd mlp mlp_sd\n")
        for sg in sigmas:
            km, ksd = ms(cf["fedkan_opt"][sg]); mm, msd = ms(cf["mlp"][sg])
            f.write("%.2f %.5f %.5f %.5f %.5f\n" % (sg, km, ksd, mm, msd))
    print("G csi-fed done")
    print("ALL STUDIES DONE -> paper/data/")


if __name__ == "__main__":
    main()
