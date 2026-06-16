#!/usr/bin/env bash
# Full scaled, multi-seed run for the GPU server. Runs the headline experiment at
# a larger scale + more seeds, the depth studies, then regenerates all figure data.
# Safe to launch in tmux. Override scale via env vars.
#
#   conda activate evokan
#   bash scripts/run_all_gpu.sh
set -u
export PYTHONUNBUFFERED=1          # stream progress lines through tee in real time
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
TS="$(date +%Y%m%d_%H%M%S)"; LOG="results/run_gpu_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

SEEDS="${SEEDS:-20}"            # headline seeds (was 10)
NODES="${NODES:-24}"           # terminals (was 10)
SLOTS="${SLOTS:-30}"           # slots per pass (was 24)
PASSES="${PASSES:-5}"          # passes (was 4)
SAMP="${SAMP:-160}"            # states per slot (was 128)
CSI="${CSI:-0.0}"              # CSI estimation-error std (0 = perfect CSI)
DEV="${DEV:-cuda}"

echo "===== EvoKAN scaled run @ ${TS} ====="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "SEEDS=$SEEDS NODES=$NODES SLOTS=$SLOTS PASSES=$PASSES SAMP=$SAMP CSI=$CSI"

echo "---- smoke + tests ----"
python scripts/smoke_opt.py || { echo "SMOKE FAIL"; exit 1; }
python -m pytest tests/ -q || echo "pytest issues (continuing)"

echo "---- [1/3] headline (scaled, multi-seed) ----"
python scripts/run_opt.py --seeds "$SEEDS" --n-nodes "$NODES" \
    --slots-per-pass "$SLOTS" --n-passes "$PASSES" --slot-samples "$SAMP" \
    --csi-sigma "$CSI" --device "$DEV" --log-every 40 --out results/opt.csv

echo "---- [2/3] depth studies (generalization/pareto/scaling/capacity) ----"
python scripts/run_studies.py

echo "---- [3/3] regenerate figure data ----"
python paper/make_figs_data.py

echo "===== DONE @ $(date +%H:%M:%S) ====="
echo "Push back: git add results/ paper/data/ && git commit -m 'results: scaled GPU run ${TS}' && git push"
