#!/usr/bin/env bash
# Run two CTGP jobs in parallel on cuda:0 and cuda:1.
# Assign different tasks / representation levels per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p results/parallel

echo "Starting metal_opt L0 on cuda:0..."
CUDA_VISIBLE_DEVICES=0 python ctgp.py \
  --device cuda:0 \
  --leaderboard \
  --class_target metal_opt \
  --repr_level 0 \
  --run_ctgp \
  --methods ctgp_1plus1 \
  --generations 300 \
  --runs 10 \
  --output_dir results/parallel/metal_opt_L0_gpu0 \
  > results/parallel/metal_opt_L0_gpu0.log 2>&1 &
PID0=$!

echo "Starting spillage L2 on cuda:1..."
CUDA_VISIBLE_DEVICES=1 python ctgp.py \
  --device cuda:0 \
  --leaderboard \
  --class_target spillage \
  --repr_level 2 \
  --run_ctgp \
  --methods ctgp_1plus1 \
  --generations 300 \
  --runs 10 \
  --output_dir results/parallel/spillage_L2_gpu1 \
  > results/parallel/spillage_L2_gpu1.log 2>&1 &
PID1=$!

echo "Jobs launched: PID cuda:0=${PID0}, PID cuda:1=${PID1}"
echo "Tail logs: tail -f results/parallel/*.log"
wait "$PID0" "$PID1"
echo "Both jobs finished."
