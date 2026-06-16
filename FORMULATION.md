# Problem formulation (S2) — Interpretable Federated KAN Surrogate for Online LEO Resource Optimization

Target: IEEE JSTSP SI "Autonomous and Evolutive Optimization in Networked AI".

## System model
A set of gateway/UAV terminals $\mathcal{N}$ serve many links over LEO satellite
backhaul. At slot $t$, terminal $n$ faces link states $s=(g, I, p_0)$: channel
gain $g$ (diverse across served links), interference $I_n(t)$ (set by the
currently serving satellite/beam), and a circuit/energy term $p_0$. It chooses
transmit power $p$ to maximize energy efficiency
$$\mathrm{EE}(p;s)=\frac{\log_2\!\big(1+\tfrac{g\,p}{1+I}\big)}{p+p_0}.$$
$\mathrm{EE}$ is quasi-concave with a unique maximizer $p^\*(s)$ solving a
transcendental stationarity condition; **no closed form** exists.

## Non-stationarity
Over an LEO pass, satellite handovers switch the interference regime $I_n(t)$
(piecewise-constant with change points), and visibility windows gate
participation. Thus the operative region of the state space and the relevant
slice of $p^\*(\cdot)$ **drift in time** -> the surrogate must adapt online.

## The problem (P1)
Learn a shared surrogate $\hat p_\theta:\,s\mapsto p$ minimizing time-averaged
regret to the oracle while respecting an uplink budget:
$$\min_{\theta,\{m_{n,t}\}}\ \frac1T\sum_t \mathbb{E}_s\Big[\mathrm{EE}(p^\*(s);s)-\mathrm{EE}(\hat p_\theta(s);s)\Big]\quad\text{s.t.}\ \sum_n \|m_{n,t}\|_0\,\beta \le B(t),$$
with federated aggregation across $\mathcal{N}$ and binary update masks $m$
(sparsification) for communication. Non-convex (KAN/MLP training), coupled by the
shared $\theta$ and the per-slot budget $B(t)$, time-varying via $I_n(t)$.

## Approach (validated)
- **KAN surrogate.** Splines approximate the smooth $p^\*(\cdot)$ at ~2x lower
  NMSE than a parameter-matched MLP, and expose **closed-form rules** (each input
  -> dominant functional form, fidelity $R^2$).
- **Online adaptation.** A drift detector on the surrogate error flags interference
  handovers; the federated model tracks the new regime.
- **Communication efficiency.** A contextual-bandit controller sets per-slot
  update sparsity against the budget $B(t)$ (acc-vs-bits Pareto).
- **Honest ablation.** Grid self-evolution (growing the spline grid on drift) does
  NOT beat a well-sized static grid in this federated streaming regime -> reported
  as a negative result, with a static grid used in the proposed method.

## Baselines / ablations
fedkan_opt (proposed), kan_full (un-sparsified KAN), kan_evolve (+grid growth,
negative ablation), mlp / mlp_prox (param-matched), linear (shows nonlinearity).

## Metrics
Optimality gap (EE regret), NMSE on $p^\*$, rule fidelity $R^2$, uplink bits,
#params, drift-detection delay, retention across recurring regimes.
