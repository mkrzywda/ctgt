#!/usr/bin/env python3
"""Split the monolithic CTGP implementation into the ctgp_lib package."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ctgp research monolith"
OUT = Path(__file__).resolve().parents[1] / "ctgp_lib"

SECTIONS = [
    ("data/jarvis.py", 97, 852, [
        "import io",
        "import json",
        "import os",
        "import time",
        "import zipfile",
        "import urllib.request",
        "import numpy as np",
        "import torch",
        "from torch_geometric.loader import DataLoader",
        "from torch_geometric.data import Data",
        "from sklearn.model_selection import train_test_split",
        "from ctgp_lib.config import *",
        "from ctgp_lib.runtime import logger, seed, JARVIS_LOADER_CACHE",
        "from ctgp_lib.device import get_device",
    ]),
    ("models/gnn.py", 853, 1011, [
        "import copy",
        "import time",
        "import numpy as np",
        "import torch",
        "import torch.nn.functional as F",
        "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score",
        "from ctgp_lib.runtime import process",
        "from ctgp_lib.device import get_device",
    ]),
    ("train/evaluate.py", 1012, 1065, [
        "import copy",
        "import time",
        "import numpy as np",
        "import torch",
        "import torch.nn.functional as F",
        "from ctgp_lib.runtime import process",
        "from ctgp_lib.device import get_device",
        "from ctgp_lib.models.gnn import GNN, evaluate_loader_metrics",
    ]),
    ("search/surrogate.py", 1066, 1247, [
        "import math",
        "import time",
        "import numpy as np",
        "import torch",
        "import torch.nn.functional as F",
        "from ctgp_lib.config import CONV_NAMES, INIT_OPTIONS, SURROGATE_FEATURE_DIM",
        "from ctgp_lib.device import get_device",
        "from ctgp_lib.train.evaluate import evaluate_loader_metrics, evaluate_solution_with_budget",
        "try:\n    import xgboost as xgb\nexcept Exception:\n    xgb = None",
        "try:\n    import lightgbm as lgb\nexcept Exception:\n    lgb = None",
    ]),
    ("genotype/concat_tree.py", 1249, 1390, [
        "import copy",
        "import math",
        "import random",
        "import numpy as np",
        "from ctgp_lib.config import *",
    ]),
    ("utils/io.py", 1396, 1413, [
        "import os",
        "import csv",
        "import pandas as pd",
    ]),
    ("search/evolution.py", 1415, 1971, [
        "import copy",
        "import logging",
        "import os",
        "import time",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "from ctgp_lib.runtime import logger",
        "from ctgp_lib.genotype.concat_tree import *",
        "from ctgp_lib.train.evaluate import evaluate_solution_with_budget, collect_final_metrics",
        "from ctgp_lib.search.surrogate import GraphSurrogateManager, extract_solution_features",
        "from ctgp_lib.utils.io import append_csv_row, save_history_csv",
    ]),
    ("utils/seed.py", 1977, 2020, [
        "import json",
        "import os",
        "import random",
        "import numpy as np",
        "import torch",
    ]),
    ("baselines/sklearn_ml.py", 2589, 2770, [
        "import numpy as np",
        "import torch",
        "import torch.nn.functional as F",
        "from sklearn.ensemble import RandomForestClassifier",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.svm import SVC",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.metrics import accuracy_score, f1_score",
        "from ctgp_lib.models.gnn import safe_roc_auc",
    ]),
]

EXPERIMENT_HEADER = """import argparse
import copy
import json
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ctgp_lib.baselines.sklearn_ml import train_baseline_mlp, train_baseline_sklearn
from ctgp_lib.config import LEADERBOARD_BENCHMARK_MAP
from ctgp_lib.data.jarvis import download_leaderboard_splits, get_jarvis_loaders
from ctgp_lib.device import get_device
from ctgp_lib.models.gnn import GNN, collect_predictions_with_jids, save_leaderboard_csv
from ctgp_lib.runtime import SAVE_DIR, logger
from ctgp_lib.search.evolution import (
    plot_search_history,
    run_ctgp_1plus1,
    run_ctgp_1plus4,
    run_ctgp_1plus10,
)
from ctgp_lib.utils.io import append_csv_row, save_history_csv
from ctgp_lib.utils.seed import build_run_seed, save_best_artifacts, set_run_seed
"""


def read_lines() -> list[str]:
    return SRC.read_text(encoding="utf-8").splitlines(keepends=True)


def extract_block(lines: list[str], start: int, end_exclusive: int) -> str:
    """Extract lines [start, end_exclusive) using 1-based line numbers."""
    return "".join(lines[start - 1 : end_exclusive - 1])


def patch_ctgp_names(text: str) -> str:
    """Align generated identifiers with the CTGP paper naming."""
    replacements = [
        ("run_cgp_1plus1_jarvis", "run_ctgp_1plus1"),
        ("run_cgp_1plus4_jarvis", "run_ctgp_1plus4"),
        ("run_cgp_1plus10_jarvis", "run_ctgp_1plus10"),
        ("cgp_surrogate_1plus10", "ctgp_surrogate_1plus10"),
        ("cgp_surrogate_1plus4", "ctgp_surrogate_1plus4"),
        ("cgp_surrogate_1plus1", "ctgp_surrogate_1plus1"),
        ("cgp_1plus10", "ctgp_1plus10"),
        ("cgp_1plus4", "ctgp_1plus4"),
        ("cgp_1plus1", "ctgp_1plus1"),
        ("--run_cgp", "--run_ctgp"),
        ("run_cgp", "run_ctgp"),
        ("cgp_methods", "ctgp_methods"),
        ("cgp_runner", "ctgp_runner"),
        ("cgp_df", "ctgp_df"),
        ("jarvis_cgp_experiments", "jarvis_ctgp_experiments"),
        ("CGP + Surrogate", "CTGP + Surrogate"),
        ("CGP methods", "CTGP methods"),
        ("CGP method", "CTGP method"),
        ("CGP mutation", "CTGP mutation"),
        ("CGP solution", "CTGP solution"),
        ("CGP Operations", "CTGP genotype operations"),
        ("| CGP+Surrogate |", "| CTGP+Surrogate |"),
        ("No CGP methods", "No CTGP methods"),
        ("Run CGP", "Run CTGP"),
        ("then CGP", "then CTGP"),
        ("# Run CGP", "# Run CTGP"),
        ('Run all methods (CGP + baselines)', 'Run all methods (CTGP + baselines)'),
        ('"""CGP 1+1', '"""CTGP 1+1'),
        ('"""CGP 1+4', '"""CTGP 1+4'),
        ('"""CGP 1+10', '"""CTGP 1+10'),
        ("description='CGP + Surrogate", "description='CTGP + Surrogate"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_device(text: str) -> str:
    text = text.replace(".to(device)", ".to(get_device())")
    text = text.replace("batch.to(device)", "batch.to(get_device())")
    text = text.replace("map_location=device", "map_location=get_device()")
    # Avoid double-replacement
    text = text.replace(".to(get_device())", ".to(get_device())")
    return text


def write_module(rel_path: str, header: list[str], body: str) -> None:
    path = OUT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    content = '"""Auto-generated from the CTGP research monolith — do not edit by hand."""\n\n'
    content += "\n".join(header) + "\n\n"
    content += patch_ctgp_names(patch_device(body))
    path.write_text(content, encoding="utf-8")


def write_config() -> None:
    lines = read_lines()
    body = extract_block(lines, 49, 58)
    leaderboard = extract_block(lines, 82, 91)
    path = OUT / "config.py"
    path.write_text(
        '"""Search-space constants and leaderboard mappings."""\n\n' + body + "\n" + leaderboard,
        encoding="utf-8",
    )


def write_runtime() -> None:
    header = """import logging
import random

import numpy as np
import torch

try:
    import psutil
except Exception:
    psutil = None
"""
    body = """logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

process = psutil.Process() if psutil is not None else None
SAVE_DIR = None
JARVIS_LOADER_CACHE = {}
"""
    (OUT / "runtime.py").write_text(header + "\n" + body, encoding="utf-8")


def write_device() -> None:
    (OUT / "device.py").write_text(
        '''"""CUDA / CPU device selection (cuda:0, cuda:1, cpu)."""

from __future__ import annotations

import torch

_device: torch.device | None = None


def set_device(device: str = "cuda:0") -> torch.device:
    """Configure the compute device used by all CTGP modules."""
    global _device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device!r} but CUDA is not available.")
        _device = torch.device(device)
    else:
        _device = torch.device(device)
    return _device


def get_device() -> torch.device:
    if _device is None:
        default = "cuda:0" if torch.cuda.is_available() else "cpu"
        return set_device(default)
    return _device
''',
        encoding="utf-8",
    )


def write_init_files() -> None:
    for sub in ("", "data", "models", "train", "search", "genotype", "baselines", "utils"):
        path = OUT / sub / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('"""CTGP library."""\n', encoding="utf-8")

    (OUT / "__init__.py").write_text(
        '"""Concat-Tree Genetic Programming for GNN architecture search on JARVIS-DFT."""\n',
        encoding="utf-8",
    )


def write_experiment(lines: list[str]) -> None:
    body = (
        extract_block(lines, 2022, 2379)
        + "\n\n"
        + extract_block(lines, 2381, 2536)
        + "\n\n"
        + extract_block(lines, 2542, 2588)
    )
    parse_args_body = extract_block(lines, 2773, 2871)
    path = OUT / "experiment.py"
    path.write_text(
        '"""Main experiment loop and CLI argument definitions."""\n\n'
        + EXPERIMENT_HEADER
        + "\n"
        + patch_ctgp_names(patch_device(body))
        + "\n\n"
        + parse_args_body,
        encoding="utf-8",
    )


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    lines = read_lines()
    OUT.mkdir(parents=True, exist_ok=True)
    write_init_files()
    write_config()
    write_runtime()
    write_device()
    for rel, start, end, header in SECTIONS:
        write_module(rel, header, extract_block(lines, start, end))
    write_experiment(lines)
    print(f"Package written to {OUT}")


if __name__ == "__main__":
    main()
