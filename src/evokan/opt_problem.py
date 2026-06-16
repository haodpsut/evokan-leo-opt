"""The networked optimization whose SOLUTION MAP the KAN learns.

At each link state s = (channel gain g, interference I, circuit/energy term p0)
a terminal chooses transmit power p in [0, Pmax] to maximize energy efficiency

    EE(p; s) = log2(1 + g p / (1 + I)) / (p + p0)         [bits/Hz/Joule]

EE(p) is quasi-concave with a unique maximizer p*(s) that solves a transcendental
stationarity condition: there is NO clean closed form, which is exactly what
justifies a learned surrogate. p*(s) is a smooth, deterministic (noise-free)
function of s, the regime where a KAN beats an MLP at matched parameters and, far
more importantly, where the learned splines can be read off as interpretable
design rules.

Non-stationarity: over an LEO pass the elevation sweep moves g, and satellite
handovers switch the interference regime I. Both shift the operative region of
s-space and hence the surrogate must track p*(.) online (drift compensation),
which is where self-evolution (grid growth) enters.
"""
from __future__ import annotations
import numpy as np
from . import orbit


def ee_utility(p, g, I, p0):
    """Energy efficiency at power p for state (g, I, p0). Broadcasts."""
    p = np.asarray(p, float)
    snr = g * p / (1.0 + I)
    return np.log2(1.0 + snr) / (p + p0)


def optimal_power(g, I, p0, Pmax=5.0, n_grid=400):
    """Oracle p* = argmax_p EE(p; s), by a dense grid search (vectorized).

    g, I, p0 are arrays of shape (N,). Returns (p_star, ee_star) shape (N,).
    """
    g = np.asarray(g, float); I = np.asarray(I, float); p0 = np.asarray(p0, float)
    pgrid = np.linspace(1e-3, Pmax, n_grid)[None, :]          # (1, G)
    snr = g[:, None] * pgrid / (1.0 + I[:, None])
    ee = np.log2(1.0 + snr) / (pgrid + p0[:, None])           # (N, G)
    j = ee.argmax(1)
    return pgrid[0, j], ee[np.arange(len(g)), j]


def interference_schedule(T, rng, levels=(0.2, 1.0, 3.0), min_dwell=8):
    """Piecewise-constant interference over the horizon; each switch = a handover
    / regime change (a true drift point). Returns (I_t (T,), switch_slots set)."""
    I = np.empty(T)
    switches = set()
    t = 0
    cur = rng.integers(0, len(levels))
    while t < T:
        dwell = int(rng.integers(min_dwell, 2 * min_dwell))
        I[t:t + dwell] = levels[cur]
        if t > 0:
            switches.add(t)
        nxt = rng.integers(0, len(levels))
        while nxt == cur and len(levels) > 1:
            nxt = rng.integers(0, len(levels))
        cur = nxt
        t += dwell
    return I[:T], switches


def pass_gain(n_slots, max_elev, snr_zenith_db=22.0, theta_min=10.0):
    """Linear channel gain proxy along an LEO pass from elevation path loss."""
    el = orbit.pass_elevation(n_slots, max_elev, theta_min)
    g_db = orbit.snr_from_elevation(el, snr_zenith_db)
    vis = el >= theta_min
    return 10.0 ** (g_db / 10.0), el, vis


def sample_states(I_now, n, rng, Pmax=5.0, g_lo=0.2, g_hi=30.0, p0_lo=0.1, p0_hi=1.0,
                  csi_sigma=0.0):
    """Draw n link states for a gateway serving many links under the CURRENT
    interference regime I_now (the drift variable). Channel gains g span a broad
    range (diverse links/sub-bands) so each slot has full g-coverage; the
    non-stationarity is the interference-regime switch I_now (LEO handover).

    csi_sigma>0 injects log-normal CSI estimation error: the oracle label uses the
    TRUE gain, but the state fed to the surrogate carries the estimate g_hat."""
    g = np.exp(rng.uniform(np.log(g_lo), np.log(g_hi), n))     # log-uniform over the range
    p0a = rng.uniform(p0_lo, p0_hi, n)
    I = np.full(n, I_now)
    pstar, eestar = optimal_power(g, I, p0a, Pmax=Pmax)         # oracle on TRUE gain
    g_obs = g * np.exp(csi_sigma * rng.standard_normal(n)) if csi_sigma > 0 else g
    X = np.stack([g_obs, I, p0a], axis=1).astype(np.float32)    # surrogate sees estimate
    return X, pstar.astype(np.float32), eestar.astype(np.float32)


# fixed standardization stats for the 3 state features (known ranges) so all
# nodes/regimes share one input scaling (federated consistency).
def standardize_states(X, stats):
    return ((X - stats["mu"]) / stats["sd"]).astype(np.float32)


def feature_stats(Pmax=5.0):
    # rough means/stds over the operating ranges (g in ~[0,30], I in [0.2,3], p0 in [0.1,1])
    return {"mu": np.array([[8.0, 1.4, 0.55]], np.float32),
            "sd": np.array([[8.0, 1.0, 0.27]], np.float32)}
