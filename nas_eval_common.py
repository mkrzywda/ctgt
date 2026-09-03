#!/usr/bin/env python3
"""Shared helpers for fair GNN baselines (fixed stack, random search, hparam tuning)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ctgp_lib.config import (
    CONV_NAMES,
    HID_MAX,
    HID_MIN,
    LEADERBOARD_BENCHMARK_MAP,
    MAX_LAYERS,
    MIN_LAYERS,
)
from ctgp_lib.data.jarvis import download_leaderboard_splits, get_jarvis_loaders
from ctgp_lib.device import get_device
from ctgp_lib.train.evaluate import collect_final_metrics, evaluate_solution_with_budget
from ctgp_lib.utils.io import append_csv_row
from ctgp_lib.utils.seed import build_run_seed, save_best_artifacts, set_run_seed

CONV_FIXED = ["GCN", "GraphSAGE", "GAT", "GIN"]

FIXED_GNN_DEFAULTS = {
    "n_layers": 3,
    "hidden": 64,
    "heads": 4,
    "dropout": 0.2,
    "lr": 1e-3,
    "init": "xavier",
    "cls_hidden": [64],
}


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--class_target",
        type=str,
        default="metal_opt",
        choices=list(LEADERBOARD_BENCHMARK_MAP.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--max_jarvis_samples", type=int, default=5000)
    parser.add_argument("--repr_level", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--leaderboard_cache_dir", type=str, default=".jarvis_lb_cache")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    return parser


def configure_device(args) -> None:
    from ctgp_lib.device import set_device

    set_device(getattr(args, "device", "cuda:0"))


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    return logging.getLogger("nas_baselines")


def run_indices(args) -> list[int]:
    if getattr(args, "run", None) is not None:
        run = int(args.run)
        if run < 1:
            raise ValueError(f"--run must be >= 1, got {run}")
        return [run]
    return list(range(1, int(args.runs) + 1))


def load_loaders(args):
    leaderboard_splits = None
    if args.leaderboard:
        bench = LEADERBOARD_BENCHMARK_MAP[args.class_target]
        leaderboard_splits = download_leaderboard_splits(bench, args.leaderboard_cache_dir)
        args.max_jarvis_samples = 0
    return get_jarvis_loaders(
        batch_size=args.batch_size,
        jarvis_dataset="dft_3d",
        class_target=args.class_target,
        class_threshold=0.01,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        max_samples=args.max_jarvis_samples,
        leaderboard_splits=leaderboard_splits,
        repr_level=args.repr_level,
    )


def loader_labels(loader) -> np.ndarray:
    ys = []
    for batch in loader:
        ys.append(batch.y.view(-1).cpu().numpy())
    if not ys:
        return np.array([], dtype=int)
    return np.concatenate(ys).astype(int)


def split_stats(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    n = int(y.size)
    if n == 0:
        return {"n": 0, "majority_acc": float("nan"), "pos_frac": float("nan")}
    counts = Counter(y.tolist())
    return {
        "n": n,
        "majority_acc": max(counts.values()) / n,
        "pos_frac": counts.get(1, 0) / n,
    }


def extra_classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict_loader(model, loader):
    model.eval()
    y_t, y_p = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(get_device())
            logits = model(batch)
            y_t.extend(batch.y.view(-1).long().cpu().tolist())
            y_p.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(y_t, dtype=int), np.asarray(y_p, dtype=int)


def make_fixed_gnn_solution(conv_type, n_layers=None, hidden=None, heads=None,
                            dropout=None, lr=None, init=None, cls_hidden=None):
    if conv_type not in CONV_NAMES:
        raise ValueError(f"Unknown conv type '{conv_type}'")
    cfg = dict(FIXED_GNN_DEFAULTS)
    if n_layers is not None:
        cfg["n_layers"] = int(n_layers)
    if hidden is not None:
        cfg["hidden"] = int(hidden)
    if heads is not None:
        cfg["heads"] = int(heads)
    if dropout is not None:
        cfg["drop"] = float(dropout)
        cfg["dropout"] = float(dropout)
    if lr is not None:
        cfg["lr"] = float(lr)
    if init is not None:
        cfg["init"] = init
    if cls_hidden is not None:
        cfg["cls_hidden"] = list(cls_hidden)

    n_layers = max(MIN_LAYERS, min(MAX_LAYERS, int(cfg["n_layers"])))
    hidden = max(HID_MIN, min(HID_MAX, int(cfg["hidden"])))
    heads = int(cfg["heads"]) if conv_type == "GAT" else 1
    drop = float(cfg.get("dropout", cfg.get("drop", 0.2)))
    return {
        "conv": [conv_type] * n_layers,
        "heads": [heads] * n_layers,
        "hidden": [hidden] * n_layers,
        "drop": drop,
        "lr": float(cfg["lr"]),
        "init": cfg["init"],
        "cls_hidden": [max(HID_MIN, min(HID_MAX, int(h))) for h in cfg["cls_hidden"]],
    }


def evaluate_solution(sol, loaders, max_epochs, patience):
    tr, vl, ts, in_f, out_f = loaders
    start = time.time()
    _, best_auc, _, best_params, model = evaluate_solution_with_budget(
        sol, tr, vl, in_f, out_f, max_epochs=max_epochs, patience=patience
    )
    val_metrics, test_metrics = collect_final_metrics(model, tr, vl, ts)
    y_val, p_val = predict_loader(model, vl if vl is not None else tr)
    y_test, p_test = predict_loader(model, ts if ts is not None else (vl if vl is not None else tr))
    val_metrics = {**val_metrics, "balanced_acc": extra_classification_metrics(y_val, p_val)["balanced_acc"]}
    test_metrics = {**test_metrics, "balanced_acc": extra_classification_metrics(y_test, p_test)["balanced_acc"]}
    return {
        "best_solution": sol,
        "best_model": model,
        "best_acc": float(val_metrics["acc"]),
        "best_auc": float(best_auc),
        "best_params": int(best_params),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_time_sec": time.time() - start,
        "surrogate_rejected_total": 0,
        "surrogate_full_evaluations": 1,
        "best_history": [],
    }


def result_row(args, method, strategy, run, run_seed, result, run_dir, extra=None):
    row = {
        "task": "graph",
        "dataset": "jarvis_dft_3d",
        "target": args.class_target,
        "method": method,
        "strategy": strategy,
        "repr_level": args.repr_level,
        "run": run,
        "seed": run_seed,
        "best_val_auc": result["val_metrics"]["auc"],
        "best_val_acc": result["val_metrics"]["acc"],
        "best_val_balanced_acc": result["val_metrics"].get("balanced_acc"),
        "best_val_f1": result["val_metrics"]["f1"],
        "best_test_auc": result["test_metrics"]["auc"],
        "best_test_acc": result["test_metrics"]["acc"],
        "best_test_balanced_acc": result["test_metrics"].get("balanced_acc"),
        "best_test_f1": result["test_metrics"]["f1"],
        "best_params": result["best_params"],
        "elapsed_time_sec": result["elapsed_time_sec"],
        "full_evaluations": result.get("surrogate_full_evaluations", 1),
        "artifact_dir": run_dir,
        "architecture_str": " -> ".join(result["best_solution"].get("conv", [])),
        "n_conv_layers": len(result["best_solution"].get("conv", [])),
        "lr": result["best_solution"].get("lr"),
        "dropout": result["best_solution"].get("drop"),
        "graph_cutoff": args.cutoff,
        "graph_max_neighbors": args.max_neighbors,
        "leaderboard": int(bool(args.leaderboard)),
    }
    if extra:
        row.update(extra)
    return row


def run_is_complete(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "best_model_info.json"))


def write_split_reference(path, loaders):
    tr, vl, ts, _, _ = loaders
    stats = {
        "train": split_stats(loader_labels(tr)),
        "val": split_stats(loader_labels(vl)),
        "test": split_stats(loader_labels(ts)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
