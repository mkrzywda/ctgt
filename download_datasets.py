#!/usr/bin/env python3
"""Download JARVIS-DFT structures and NIST leaderboard train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ctgp_lib.config import LEADERBOARD_BENCHMARK_MAP
from ctgp_lib.data.jarvis import download_leaderboard_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch JARVIS Leaderboard split JSON files")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(LEADERBOARD_BENCHMARK_MAP.keys()),
        choices=list(LEADERBOARD_BENCHMARK_MAP.keys()),
        help="Classification tasks to download (default: all seven paper benchmarks)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".jarvis_lb_cache",
        help="Directory for cached leaderboard JSON splits",
    )
    parser.add_argument(
        "--prefetch_jarvis",
        action="store_true",
        help="Also download the JARVIS dft_3d Figshare archive (large; required before training)",
    )
    return parser.parse_args()


def prefetch_jarvis_dft3d() -> None:
    try:
        from jarvis.db.figshare import data as jarvis_data
    except ImportError as exc:
        raise SystemExit("Install jarvis-tools first: pip install jarvis-tools") from exc
    print("Downloading JARVIS dft_3d dataset via jarvis-tools (this may take a while)...")
    rows = jarvis_data("dft_3d")
    print(f"JARVIS dft_3d ready: {len(rows)} structures cached by jarvis-tools.")


def main() -> None:
    args = parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        benchmark = LEADERBOARD_BENCHMARK_MAP[task]
        print(f"Downloading splits for {task} ({benchmark})...")
        splits = download_leaderboard_splits(benchmark, str(cache))
        for split in ("train", "val", "test"):
            print(f"  {split}: {len(splits[split])} structures")

    if args.prefetch_jarvis:
        prefetch_jarvis_dft3d()

    print(f"\nDone. Leaderboard splits cached in: {cache.resolve()}")


if __name__ == "__main__":
    main()
