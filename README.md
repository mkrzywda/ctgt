# CTGP — Concat-Tree Genetic Programming for GNN Architecture Search

Code release accompanying the paper *Concat-Tree Genetic Programming Neural Architecture Search for Efficient Materials Property Classification toward High-Throughput Screening*.

This directory contains a modular Python implementation of **CTGP** (Concat-Tree GP NAS) for binary classification of DFT-derived materials properties in the JARVIS-DFT database, together with fair GNN baselines and dataset utilities.

## Directory layout

```
code/
├── ctgp.py                 # Main entry point — CTGP evolutionary search
├── run_fixed_gnn.py        # Fixed-stack GNN baselines (GCN, GraphSAGE, GAT, GIN)
├── run_random_search.py    # Budget-matched random architecture search (301 evals)
├── download_datasets.py    # Download NIST leaderboard splits (+ optional JARVIS prefetch)
├── nas_eval_common.py      # Shared loaders, metrics, and evaluation helpers
├── requirements.txt
├── setup_env.sh            # Create virtualenv and install dependencies
├── config/
│   ├── datasets.yaml       # Seven classification tasks and binarisation thresholds
│   └── repr_levels.yaml    # Node representation levels L0–L3
├── ctgp_lib/               # Core library (split from the research monolith)
│   ├── config.py           # Search-space constants
│   ├── device.py           # cuda:0 / cuda:1 / cpu selection
│   ├── data/jarvis.py      # JARVIS graph construction and leaderboard splits
│   ├── genotype/concat_tree.py
│   ├── models/gnn.py
│   ├── search/evolution.py
│   ├── search/surrogate.py
│   ├── train/evaluate.py
│   ├── baselines/sklearn_ml.py
│   └── experiment.py       # Main experiment loop
└── scripts/
    ├── run_ctgp_example.sh
    └── run_parallel_gpus.sh
```

## Quick start

### 1. Environment

```bash
cd code
bash setup_env.sh
source .venv-ctgp/bin/activate
```

### 2. Download data

Leaderboard train/validation/test JID splits (required for paper comparisons):

```bash
python download_datasets.py
```

Prefetch the full JARVIS `dft_3d` crystal database (large download, needed before training):

```bash
python download_datasets.py --prefetch_jarvis
```

### 3. Run CTGP search

Single GPU (`cuda:0`):

```bash
python ctgp.py \
  --device cuda:0 \
  --leaderboard \
  --class_target metal_opt \
  --repr_level 0 \
  --run_ctgp \
  --methods ctgp_1plus1 ctgp_surrogate_1plus1 \
  --generations 300 \
  --runs 10 \
  --output_dir results/metal_opt_L0
```

Second GPU (`cuda:1`):

```bash
python ctgp.py \
  --device cuda:1 \
  --leaderboard \
  --class_target spillage \
  --repr_level 2 \
  --run_ctgp \
  --methods ctgp_1plus4 \
  --generations 300 \
  --runs 10 \
  --output_dir results/spillage_L2
```

Run two jobs in parallel (one per GPU):

```bash
bash scripts/run_parallel_gpus.sh
```

### 4. Fair GNN baselines

```bash
python run_fixed_gnn.py \
  --device cuda:0 \
  --leaderboard \
  --class_target metal_opt \
  --repr_level 0 \
  --runs 10 \
  --output_dir results/fixed_gnn/metal_opt_L0

python run_random_search.py \
  --device cuda:0 \
  --leaderboard \
  --class_target metal_opt \
  --repr_level 0 \
  --n_candidates 301 \
  --runs 10 \
  --output_dir results/random_search/metal_opt_L0
```

## Classification tasks

| Key | JARVIS benchmark | Positive class |
|-----|------------------|----------------|
| `metal_opt` | `dft_3d_optb88vdw_bandgap` | metal (gap ≤ 0.01 eV) |
| `metal_mbj` | `dft_3d_mbj_bandgap` | metal |
| `magnetic` | `dft_3d_magmom_oszicar` | magnetic (|μ_B| ≥ 0.05) |
| `slme` | `dft_3d_slme` | high SLME (≥ 10%) |
| `spillage` | `dft_3d_spillage` | high spillage (≥ 0.1) |
| `seebeck_p` | `dft_3d_p_Seebeck` | high p-type Seebeck (≥ 100 µV/K) |
| `powerfactor_n` | `dft_3d_n_powerfact` | high n-type power factor |

## Representation levels

| Level | Node features | Dimension |
|-------|---------------|-----------|
| L0 | atomic number Z | 1 |
| L1 | CGCNN atom embeddings | 92 |
| L2 | L1 + Gaussian RBF bond distances | 102 |
| L3 | L2 + bond-angle statistics | 106 |

Graph topology is fixed: k-NN, 8 Å cutoff, max 12 neighbours (see paper Table 4).

## CTGP search strategies

| CLI method name | Description |
|-----------------|-------------|
| `ctgp_1plus1` | Elitist (1+1), ~301 full evaluations |
| `ctgp_surrogate_1plus1` | (1+1) + pairwise MLP surrogate filter |
| `ctgp_1plus4` | Small population (1+4), ~1,201 evaluations |
| `ctgp_surrogate_1plus4` | (1+4) + surrogate filter |

## Rebuilding the library

The `ctgp_lib/` package is generated from the research monolith `ctgp research monolith` at the repository root:

```bash
python scripts/build_package.py
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{krzywda2026concattree,
  author    = {Krzywda, Maciej and Wermiński, Mariusz and
               Łukasik, Szymon and Gandomi, Amir H.},
  title     = {Concat-tree genetic programming for graph neural
               architecture search: Efficient materials property
               classification toward high-throughput screening},
  journal   = {Computer Methods in Materials Science},
  year      = {2026},
  volume    = {26},
  number    = {4},
  pages     = {5--33},
  doi       = {10.7494/cmms.2026.4.1050},
  url       = {https://doi.org/10.7494/cmms.2026.4.1050},
  month     = nov,
  publisher = {AGH University Press}
}
```

## License

See the repository root `LICENSE` file.
