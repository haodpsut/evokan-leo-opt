"""Interpretability: read off closed-form design rules from a trained KAN.

For each univariate edge function of the first KAN layer we fit a small library of
candidate forms (linear, quadratic, log, sqrt, reciprocal) and keep the best by
R^2. The reported rule + fidelity is the deliverable an MLP cannot provide: it
turns the learned optimal-power policy into human-readable expressions whose
fidelity we quantify.
"""
from __future__ import annotations
import numpy as np
import torch

_LIB = {
    "linear":     lambda x: x,
    "quadratic":  lambda x: x ** 2,
    "log1p":      lambda x: np.log1p(np.abs(x)),
    "sqrt":       lambda x: np.sqrt(np.abs(x)),
    "recip":      lambda x: 1.0 / (1.0 + np.abs(x)),
}


def _fit_form(x, y):
    """Best single-basis fit y ~ a*f(x)+b; return (name, R2, a, b)."""
    best = ("const", -np.inf, 0.0, float(np.mean(y)))
    for name, f in _LIB.items():
        fx = f(x)
        A = np.c_[fx, np.ones_like(fx)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss = 1.0 - np.sum((y - pred) ** 2) / (np.var(y) * len(y) + 1e-12)
        if ss > best[1]:
            best = (name, float(ss), float(coef[0]), float(coef[1]))
    return best


@torch.no_grad()
def extract_rules(kan_model, n=400, x_range=(-2.0, 2.0)):
    """Per input feature of layer 0, fit the dominant functional form.

    Returns list of dicts: {feature, form, r2, a, b} and the mean R^2 (fidelity).
    """
    layer = kan_model.layers[0] if hasattr(kan_model, "layers") else kan_model.head.layers[0]
    in_features = layer.in_features
    dev = next(kan_model.parameters()).device      # build probes on the model's device
    xs = torch.linspace(x_range[0], x_range[1], n, device=dev)
    rules, r2s = [], []
    for i in range(in_features):
        # probe edge i->(sum over outputs) by varying feature i, others at 0
        X = torch.zeros(n, in_features, device=dev)
        X[:, i] = xs
        y = layer(X).sum(dim=1).cpu().numpy()   # aggregate response to feature i
        name, r2, a, b = _fit_form(xs.cpu().numpy(), y)
        rules.append({"feature": i, "form": name, "r2": round(r2, 3),
                      "a": round(a, 4), "b": round(b, 4)})
        r2s.append(r2)
    return rules, float(np.mean(r2s))
