"""Deep-unfolding learn-to-optimize baseline.

UnfoldedEE unrolls K projected-gradient-ascent steps on the energy-efficiency
objective EE(p;s) with learnable step sizes (and a learnable initial power). Unlike
the KAN/MLP/linear surrogates, which are model-free and learn the map only from
oracle labels, this baseline is model-based: it embeds the analytic EE gradient.
It is the standard learn-to-optimize / deep-unfolding comparator
(after Balatsoukas-Stimming and Studer); we include it as a strong, physics-aware
reference. It is tiny by construction (K+1 parameters), so it is not parameter-
matched to the surrogates; that is expected for an unfolded solver.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import opt_problem as op


class UnfoldedEE(nn.Module):
    def __init__(self, K=15, Pmax=5.0, stats=None):
        super().__init__()
        self.K = K
        self.Pmax = Pmax
        self.log_alpha = nn.Parameter(torch.full((K,), -1.0))  # step sizes via softplus (small init)
        self.p_init = nn.Parameter(torch.tensor(0.5))
        st = stats or op.feature_stats()
        self.register_buffer("mu", torch.tensor(st["mu"][0], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(st["sd"][0], dtype=torch.float32))
        self._ln2 = float(np.log(2.0))

    def forward(self, Xstd):
        x = Xstd * self.sd + self.mu                           # de-standardize -> (g, I, p0)
        g = x[:, 0].clamp_min(1e-3); I = x[:, 1].clamp_min(0.0); p0 = x[:, 2].clamp_min(1e-3)
        a = g / (1.0 + I)
        p = self.p_init.clamp(1e-3, self.Pmax).expand_as(g)
        for k in range(self.K):
            f = torch.log1p(a * p) / self._ln2
            fp = a / ((1.0 + a * p) * self._ln2)
            dEE = (fp * (p + p0) - f) / ((p + p0) ** 2)        # exact EE gradient
            # bounded, direction-following step (stable across heterogeneous gradient scales)
            step = F.softplus(self.log_alpha[k]) * torch.tanh(dEE)
            p = (p + step).clamp(1e-3, self.Pmax)
        return p.unsqueeze(1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
