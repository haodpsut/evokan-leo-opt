"""Method presets for the LEO resource-optimization surrogate study.

Framing (validated): the KAN surrogate beats a param-matched MLP ~2x on the
optimal-power solution map and yields interpretable closed-form rules. Grid
self-evolution does NOT help in this federated streaming regime and is reported
as an honest negative ablation.

  fedkan_opt  PROPOSED: KAN surrogate, federated, bandit-controlled update
              SPARSIFICATION for communication efficiency, fixed grid.
  kan_full    KAN, full (un-sparsified) updates -> accuracy reference / high comm.
  kan_evolve  ABLATION: PROPOSED + grid self-evolution (shown not to help).
  mlp         param-matched MLP surrogate (architecture baseline).
  mlp_prox    MLP + FedProx.
  linear      linear least-squares surrogate (shows the map is nonlinear).
"""
from .experiment_opt import Config


def make_config(method: str, **overrides) -> Config:
    presets = {
        "fedkan_opt": dict(model_type="kan", controller="bandit", evolve_enabled=False),
        "kan_full":   dict(model_type="kan", controller="fixed",  evolve_enabled=False),
        "kan_evolve": dict(model_type="kan", controller="bandit", evolve_enabled=True),
        "mlp":        dict(model_type="mlp", controller="fixed",  evolve_enabled=False),
        "mlp_prox":   dict(model_type="mlp", controller="fixed",  evolve_enabled=False, mu=0.01),
        "linear":     dict(model_type="linear", controller="fixed", evolve_enabled=False),
    }
    if method not in presets:
        raise ValueError(f"unknown method {method}; choices={list(presets)}")
    return Config(**{**presets[method], **overrides})


METHODS = ["fedkan_opt", "kan_full", "kan_evolve", "mlp", "mlp_prox", "linear"]
