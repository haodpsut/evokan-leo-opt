# EvoKAN-LEO-Opt

Code for the IEEE JSTSP Special Issue paper *"Interpretable Federated
Kolmogorov-Arnold Surrogates for Online Resource Optimization over
Non-Stationary LEO Links"* (SI: Autonomous and Evolutive Optimization in
Networked AI).

A federated **Kolmogorov-Arnold Network (KAN)** learns the *solution map* of an
energy-efficiency transmit-power optimization over LEO links: state
`(channel gain, interference, circuit power) -> optimal power p*`. The optimal
policy has no closed form and drifts as the satellite passes and interference
regimes switch (handovers). The KAN surrogate:

- predicts `p*` with **~2x lower error than a parameter-matched MLP**,
- yields **closed-form, human-readable design rules** extracted from its splines
  (an MLP cannot),
- adapts **online** as the interference regime drifts, with a drift detector,
- is **communication-efficient**: a contextual-bandit controller sparsifies
  federated updates against an uplink budget.

Honest ablation: grid **self-evolution** (growing the spline grid on drift) does
**not** improve over a well-sized static grid in this federated streaming regime;
we report this as a negative result.

## Quick start
```bash
pip install -r requirements.txt
python scripts/smoke_opt.py     # SMOKE PASS
python -m pytest tests/ -q      # tests pass
python scripts/run_opt.py --seeds 5 --out results/opt.csv   # ~minutes on CPU
```

## Layout
```
src/evokan/  opt_problem (oracle+LEO states), orbit, kan, drift, controller,
             fed, experiment_opt, symbolic, methods
scripts/     smoke_opt.py, run_opt.py
tests/       test_opt.py
```
Authors: Phuc Hao Do (DAU/CAIRA, first + corresponding), Huu Phu Le (DAU/CAIRA),
Truong Duy Dinh (PTIT, funding).
