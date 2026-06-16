# run.md

Models are tiny (hundreds of params); the whole study runs in minutes on CPU.
GPU is optional.

## Setup
```bash
pip install -r requirements.txt        # or conda env create -f environment.yml
python scripts/smoke_opt.py            # expect SMOKE PASS
python -m pytest tests/ -q             # expect tests pass
```

## Full study (5 seeds, all methods)
```bash
python scripts/run_opt.py --seeds 5 --out results/opt.csv
git add results/ && git commit -m "results: opt 5-seed" && git push
```

## Outputs
- `results/opt.csv` : one row per (method, seed) with
  avg_gap, final_gap, avg_nmse, final_nmse, forgetting, detection_delay,
  total_bits, final_grid, params, rule_r2
- `results/logs/*.npz` : per-slot time series (gap, nmse, bits, grid, drift_flags)

## Story to verify (report honestly)
- KAN methods (`fedkan_opt`, `kan_full`) << MLP on `final_nmse` (target ~2x) and
  KAN gives `rule_r2` ~0.85+ (MLP has none) -> the interpretable-surrogate win.
- `fedkan_opt` uses fewer `total_bits` than `kan_full` at a modest accuracy cost
  -> the communication Pareto knob (bandit sparsification).
- `linear` is much worse -> the map is genuinely nonlinear.
- `kan_evolve` does NOT beat `kan_full`/`fedkan_opt` -> grid self-evolution is a
  negative result in this regime (report transparently).
- Wilcoxon over seeds for the KAN-vs-MLP NMSE gap; mean +/- std.
```
