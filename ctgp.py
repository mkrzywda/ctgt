#!/usr/bin/env python3
"""CTGP — Concat-Tree Genetic Programming NAS for JARVIS-DFT GNN classification.

Main entry point referenced in the paper (Section 3.6, Code and data availability).
Runs elitist (1+1) and small-population (1+4) search, with optional surrogate
filtering, across seven JARVIS Leaderboard binary classification tasks and
representation levels L0–L3.

Examples
--------
# Single task on GPU 0 (paper protocol: leaderboard splits, 300 generations)
python ctgp.py --device cuda:0 --leaderboard --class_target metal_opt \\
    --repr_level 0 --run_ctgp --generations 300 --runs 10 \\
    --output_dir results/metal_opt_L0

# Same task on GPU 1
python ctgp.py --device cuda:1 --leaderboard --class_target spillage \\
    --repr_level 2 --run_ctgp --methods ctgp_1plus1 ctgp_surrogate_1plus1 \\
    --output_dir results/spillage_L2

# Classical ML baselines only (fast smoke test)
python ctgp.py --device cpu --leaderboard --class_target metal_opt \\
    --run_baselines --runs 1 --generations 5 --output_dir results/smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pop_device_arg(argv: list[str]) -> tuple[str, list[str]]:
    device = "cuda:0"
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
            continue
        if token.startswith("--device="):
            device = token.split("=", 1)[1]
            i += 1
            continue
        out.append(token)
        i += 1
    return device, out


def main() -> None:
    device, rest = _pop_device_arg(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]

    from ctgp_lib.device import set_device
    from ctgp_lib.experiment import parse_args, run_experiments

    set_device(device)
    args = parse_args()
    if not hasattr(args, "device"):
        args.device = device
    run_experiments(args)


if __name__ == "__main__":
    main()
