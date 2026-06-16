"""Correctness tests for the optimization-surrogate pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from evokan import opt_problem as op
from evokan.kan import KANClassifier
from evokan import fed
from evokan.symbolic import extract_rules


def test_oracle_is_maximizer():
    g = np.array([5.0]); I = np.array([1.0]); p0 = np.array([0.3])
    pstar, eestar = op.optimal_power(g, I, p0)
    # EE at p* must be >= EE at random nearby powers
    for p in [0.1, 0.5, 1.0, 2.0, 4.0]:
        assert eestar[0] >= op.ee_utility(np.array([p]), g, I, p0)[0] - 1e-9


def test_kan_learns_solution_map():
    rng = np.random.default_rng(0)
    g = rng.uniform(0.2, 30, 3000); I = rng.uniform(0.2, 3, 3000); p0 = rng.uniform(0.1, 1, 3000)
    y, _ = op.optimal_power(g, I, p0)
    X = np.stack([g, I, p0], 1).astype("float32")
    stats = op.feature_stats(); Xs = op.standardize_states(X, stats)
    idx = np.arange(len(y)); rng.shuffle(idx); tr, te = idx[:2400], idx[2400:]
    m = KANClassifier(3, (8,), 1, grid_size=6)
    fed.local_train(m, Xs[tr], y[tr], epochs=60, lr=0.01, task="reg")
    assert fed.nmse(m, Xs[te], y[te]) < 0.05      # learns the map well


def test_grid_extension_preserves():
    torch.manual_seed(0)
    m = KANClassifier(3, (6,), 1, grid_size=5)
    x = torch.randn(16, 3) * 0.5
    before = m(x).detach()
    m.evolve(9)
    assert torch.allclose(before, m(x).detach(), atol=1e-2)


def test_fed_aggregate_moves_and_bits():
    torch.manual_seed(0)
    g = KANClassifier(3, (6,), 1, grid_size=5)
    clients = []
    for _ in range(3):
        c = fed.clone_model(g)
        X = np.random.randn(64, 3).astype("float32"); y = np.random.rand(64).astype("float32")
        fed.local_train(c, X, y, epochs=1, task="reg")
        clients.append({"model": c, "weight": 64, "keep": 0.5})
    bits, _ = fed.fed_aggregate(g, clients)
    assert bits > 0


def test_symbolic_returns_rules():
    m = KANClassifier(3, (6,), 1, grid_size=6)
    rules, fid = extract_rules(m)
    assert len(rules) == 3 and 0.0 <= fid <= 1.0
