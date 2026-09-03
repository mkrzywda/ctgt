#!/usr/bin/env bash
# Example: run CTGP (1+1) on spillage L2 using GPU 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DEVICE="${DEVICE:-cuda:0}"
TARGET="${TARGET:-spillage}"
LEVEL="${LEVEL:-2}"
OUTPUT="${OUTPUT:-results/${TARGET}_L${LEVEL}_1plus1}"

python ctgp.py \
  --device "${DEVICE}" \
  --leaderboard \
  --class_target "${TARGET}" \
  --repr_level "${LEVEL}" \
  --run_ctgp \
  --methods ctgp_1plus1 \
  --generations 300 \
  --runs 10 \
  --output_dir "${OUTPUT}"
