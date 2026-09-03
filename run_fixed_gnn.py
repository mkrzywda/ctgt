#!/usr/bin/env python3
"""Fixed-stack GNN baselines (GCN, GraphSAGE, GAT, GIN) on identical JARVIS inputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nas_eval_common as common
from ctgp_lib.utils.io import append_csv_row
from ctgp_lib.utils.seed import build_run_seed, save_best_artifacts, set_run_seed


def parse_args():
    parser = common.add_shared_args(
        argparse.ArgumentParser(description="Fixed GNN baselines for CTGP paper comparisons")
    )
    parser.add_argument("--models", nargs="+", default=common.CONV_FIXED, choices=common.CONV_FIXED)
    parser.add_argument("--n_layers", type=int, default=common.FIXED_GNN_DEFAULTS["n_layers"])
    parser.add_argument("--hidden", type=int, default=common.FIXED_GNN_DEFAULTS["hidden"])
    parser.add_argument("--heads", type=int, default=common.FIXED_GNN_DEFAULTS["heads"])
    parser.add_argument("--dropout", type=float, default=common.FIXED_GNN_DEFAULTS["dropout"])
    parser.add_argument("--lr", type=float, default=common.FIXED_GNN_DEFAULTS["lr"])
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
        "Fixed GNN | test majority acc=%.3f | device=%s",
        stats["test"]["majority_acc"],
        args.device,
    )

    for conv_type in args.models:
        method = f"fixed_gnn_{conv_type.lower()}"
        for run in common.run_indices(args):
            run_seed = build_run_seed(args.seed, f"{args.class_target}_L{args.repr_level}", method, run)
            set_run_seed(run_seed)
            run_dir = os.path.join(history_root, method, f"run_{run}")
            os.makedirs(run_dir, exist_ok=True)
            if args.skip_existing and common.run_is_complete(run_dir):
                logger.info("Skip existing %s run %d", method, run)
                continue

            sol = common.make_fixed_gnn_solution(
                conv_type,
                n_layers=args.n_layers,
                hidden=args.hidden,
                heads=args.heads,
                dropout=args.dropout,
                lr=args.lr,
            )
            result = common.evaluate_solution(sol, loaders, args.max_epochs, args.patience)
            save_best_artifacts(run_dir, result, "dft_3d", method, run)
            row = common.result_row(
                args, method, "fixed_gnn", run, run_seed, result, run_dir,
                extra={"test_majority_acc": stats["test"]["majority_acc"]},
            )
            append_csv_row(detailed_csv, row)
            logger.info(
                "  %s run %d | test_acc=%.4f | auc=%.4f",
                method, run, row["best_test_acc"], row["best_test_auc"],
            )


if __name__ == "__main__":
    main()
