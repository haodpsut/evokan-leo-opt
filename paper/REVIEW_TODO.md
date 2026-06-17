# Review findings to apply (3 verification subagents, 2026-06-17)

Paused before applying. Resume = work through this list, recompile, visual-sweep, push.

## A. MATH / FORMULAS (agent: math)
No hard math errors (stationarity eq, quasi-concavity proof, kappa* derivation, complexity all verified correct vs code). Fix notation clashes:
- [FIX-1, most impactful] Symbol `\rho` used twice: SII-B defines ABSOLUTE regret `EE(p*)-EE(p)`, but Metrics + code (`_utility_gap`) use RELATIVE `[EE(p*)-EE(p)]/EE(p*)`. Reported numbers are the relative one. -> redefine rho as relative in SII-B (line ~308), or use two distinct symbols.
- [FIX-2] `\lambda` clash: dual price (eq:outer, dual-price view) vs Page-Hinkley param in params table ("lambda=0.05"). Rename detector param to lambda_PH or alpha_PH.
- [FIX-3] `B` clash: uplink budget B(t) vs batch size in complexity ("O(E B |theta|)"). Rename batch to b or B_loc.
- [nit] eq:p1 (line ~342): state explicitly sum_n w_{n,t}=1 (else theta_{t-1} term doesn't telescope; code normalizes w_n/wsum).
- [nit] soften "the bandit estimates this price lambda_t" (line ~1026): code uses a FIXED comm_lambda in the reward; bandit learns the keep-fraction that optimizes the fixed-price Lagrangian, not lambda itself. Reword.

## B. BIBLIOGRAPHY (agent: bib)
All 28 entries real; 4 earlier fixes confirmed correct. Items:
- [FIX] li2020fedprox author order -> canonical MLSys lists **Tian Li FIRST** (revert the earlier Sahu-first change): `author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia}`.
- [DEAD, uncited -> remove or cite] `bifet2007adwin` (ADWIN; paper uses Page-Hinkley, so remove), `vallado2006sgp4` (SGP4; orbit is analytic now, so remove), `lu2020amc` (real+correct but uncited -> either cite in intro physical-layer-DL sentence or remove). IEEEtran drops uncited silently, but clean them up.
- [nit] shi2023kanfl key name implies "KAN-FL" but it's a generic FL-comm-efficiency survey; metadata + usage correct -> cosmetic, optional rename.

## C. FIGURES / TABLES (agent: figures) -- includes "make eye-catching"
### Correctness (fix first)
- [ERROR-1] `fig:ts` (single-seed time series): `ymax=0.6` CLIPS early transients (data up to 2.71); ~12-19% of each curve cut off. Fix: `ymode=log` OR start x>=~10 OR raise ymax. Caption currently over-states.
- [ERROR-2] `fig:conv` (convergence, full-width): NO `ymax` -> auto-scales to ~4.3 (band max), so the steady-state KAN-vs-MLP gap the caption describes is an invisible sliver. Fix: `ymax~0.6` after burn-in, or log-y, or zoomed inset.
- [nit] `tab:rules`/macro `\ruleGrtwo=0.36` vs data `rules.txt` 0.350 -> set 0.35. And g "form" labeled `nonlinear` while data best-single-basis was `linear` (R2 0.35); relabel column "characterization" or footnote so a reader cross-checking rules.txt isn't confused.
- [nit] `fig:bar` never `\ref`'d -> add a sentence referencing it or merge into Table I / Fig pareto.
- [nit] dead data files: `csi.dat` (non-fed CSI, no figure), `gen.dat` (only feeds Table II) -- fine.

### Aesthetics (the "vẽ đẹp hơn" ask) -- HIGH PRIORITY per user
- [GLOBAL] One colorblind-safe palette, LOCKED per method across ALL figures (currently inconsistent: conv/ts have kan_full=blue/ours=teal, csifed has ours=blue/mlp=red, pareto2 sweep=blue). Suggested Okabe-Ito: ours/fedkan_opt=blue(0,114,178) mark=*, kan_full=green(0,158,115) square*, kan_evolve=orange(230,159,0) diamond*, mlp=vermillion(213,94,0) triangle*, mlp_prox=purple(204,121,167), linear/unfold=gray. Make ours=blue (hero) everywhere.
- [GLOBAL] preamble `\pgfplotsset{every axis/.append style={grid=major, grid style={gray!25,very thin}, line width=1pt, mark size=2.5pt, tick label style={font=\footnotesize}, label style={font=\small}, legend style={font=\scriptsize,draw=gray!50,fill=white,fill opacity=0.85,text opacity=1,rounded corners}, legend cell align=left}}`. Replace `grid=both` -> `grid=major`.
- [Fig1 system] node distance 6-8mm, inner sep 4pt; fills inner=blue!8/outer=orange!15/server=gray!10; arrows `>=Stealth`; broadcast arrow dashed gray; add dashed group boxes "terminal n (inner)" and "federation (outer)".
- [Fig2 pareto scatter] add "better" arrow (down-left), callout box on ours point ("half uplink, beats MLP"), per-point palette colors, individual label anchors for the right-side cluster (kan_full/mlp/mlp_prox), ymax 0.10.
- [Fig3 bar] `nodes near coords` value labels; style error bars (gray, mark=-); KAN bars vs MLP bars in palette; `\ref` it.
- [Fig4 conv] (after ymax fix) band `fill=blue!12,opacity=0.5`; vertical dashed regime-change guides + annotate one "regime switch"; legend columns=3; ours thickest line.
- [Fig5 ts] (after fix) prefer `ymode=log` so transient+steady both visible; regime guides; palette consistent.
- [Fig6 rules] per-panel R^2 annotation (0.35/0.88/0.99); solid=blue spline / dashed=vermillion fit + one shared legend; ylabel only on left panel; consider identical y-range across panels.
- [Fig7 pareto2 sweep] mark+label the knee (kappa~0.5), "better" arrow, bandit point as big star "lands near knee automatically", consider `ymode=log` (sweep spans 0.05->1.64).
- [Fig8 scaling+capacity] (a) ymin 0.06 ymax 0.09 to show saturation knee; (b) `ymode=log` for the monotone decay; annotate "(a) federated saturates" vs "(b) capacity helps (centralized)".
- [Fig9 regime] value labels (tiny); legend out of plot (above, columns=3); palette; bar width 8pt.
- [Fig10 csifed] `ymode=log` or zoom inset on sigma in [0,1] (sigma=2 point 1.17 dwarfs the 0.07-vs-0.11 crossover the caption is about); shade sigma>=0.5 region "MLP comparable/better (boundary)"; +-std as fill-between bands.
- [WIDTH] keep figure* for conv + rules; consider pairing pareto(Fig2)+pareto2(Fig7) side-by-side in one figure*.

## Order to apply on resume
1. Math notation fixes (rho/lambda/B) + 2 nits.
2. Bib: revert fedprox author order; remove bifet2007adwin + vallado2006sgp4 (+ decide lu2020amc).
3. Figure correctness: fig:ts + fig:conv y-axis; ruleGrtwo 0.35 + relabel.
4. Aesthetic overhaul: preamble palette + per-figure styling/annotations.
5. Recompile, visual-sweep ALL figures (render PNG, look), verify 0 undefined/overfull, push.
