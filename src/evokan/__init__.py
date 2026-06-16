"""EvoKAN: Evolutive Kolmogorov-Arnold surrogates for autonomous resource
optimization over non-stationary LEO links.

  opt_problem.py    - the energy-efficiency power-allocation problem + oracle +
                      LEO state/interference generation (the solution map to learn).
  orbit.py          - LEO pass geometry (elevation -> channel gain, visibility).
  kan.py            - efficient-KAN B-spline layer + grid-extension (self-evolution),
                      param-matched MLP. Used here as regressors (out dim 1).
  drift.py          - online drift detectors on the surrogate-error stream.
  controller.py     - Fixed / DualThreshold / LinUCB-bandit evolve+compress control.
  fed.py            - sparse FedAvg/FedProx, MSE local training, NMSE eval.
  experiment_opt.py - federated online surrogate-learning loop over an LEO pass.
  symbolic.py       - extract interpretable closed-form rules from KAN splines.
  methods.py        - method presets (evokan / baselines / ablations).
"""
__version__ = "0.1.0"
