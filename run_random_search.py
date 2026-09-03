#!/usr/bin/env python3
"""Random architecture search baseline (301 evaluations, matched to CTGP 1+1 budget)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nas_eval_common as common
from ctgp_lib.genotype.concat_tree import _gp_to_solution, _random_gp_program
from ctgp_lib.utils.io import append_csv_row, save_history_csv
from ctgp_lib.utils.seed import build_run_seed, save_best_artifacts, set_run_seed


def parse_args():
    parser = common.add_shared_args(
        argparse.ArgumentParser(description="Random search NAS baseline for CTGP comparisons")
    )
    parser.add_argument("--n_candidates", type=int, default=301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common.configure_device(args)
    logger = common.setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    detailed_csv = os.path.join(args.output_dir, "detailed_results.csv")
    history_root = os.path.join(args.output_dir, "run_artifacts")

    loaders = common.load_loaders(args)
    stats = common.write_split_reference(os.path.join(args.output_dir, "split_stats.json"), loaders)
    logger.info(
        "Random search | n_candidates=%d | majority acc=%.3f | device=%s",
        args.n_candidates, stats["test"]["majority_acc"], args.device,
    )

    method = "random_search"
    for run in common.run_indices(args):
        run_seed = build_run_seed(args.seed, f"{args.class_target}_L{args.repr_level}", method, run)
        set_run_seed(run_seed)
        run_dir = os.path.join(history_root, method, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        if args.skip_existing and common.run_is_complete(run_dir):
            logger.info("Skip existing random_search run %d", run)
            continue

        start = time.time()
        history = []
        best = None
        for k in range(1, args.n_candidates + 1):
            sol = _gp_to_solution(_random_gp_program())
            cand = common.evaluate_solution(sol, loaders, args.max_epochs, args.patience)
            val_auc = float(cand["val_metrics"]["auc"])
            history.append({
                "candidate": k,
                "val_auc": val_auc,
                "test_acc": cand["test_metrics"]["acc"],
                "params": cand["best_params"],
            })
            if best is None or val_auc > float(best["val_metrics"]["auc"]):
                best = cand
                logger.info("  new best @%d val_auc=%.4f test_acc=%.4f", k, val_auc, cand["test_metrics"]["acc"])

        best["elapsed_time_sec"] = time.time() - start
        best["surrogate_full_evaluations"] = args.n_candidates
        save_history_csv(os.path.join(run_dir, "history.csv"), history)
        save_best_artifacts(run_dir, best, "dft_3d", method, run)
        row = common.result_row(args, method, "random_search", run, run_seed, best, run_dir)
        append_csv_row(detailed_csv, row)


if __name__ == "__main__":
    main()
