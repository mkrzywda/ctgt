#!/usr/bin/env bash
# Create a Python virtual environment for CTGP experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv-ctgp}"

echo "Creating virtual environment in ${VENV_DIR}..."
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools

# Detect torch / CUDA for PyG wheels
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo '2.1.0')"
if python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null | grep -q .; then
  CUDA_TAG="cu$(python -c 'import torch; v=torch.version.cuda; print(v.replace(".","") if v else "118")')"
else
  CUDA_TAG="cpu"
fi

echo "Installing PyTorch (adjust if you need a specific CUDA build)..."
pip install "torch>=2.1.0" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
CUDA_VER="$(python -c 'import torch; v=torch.version.cuda; print(v if v else "cpu")')"
echo "Installing PyTorch Geometric extensions for torch=${TORCH_VER}, cuda=${CUDA_VER}..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" || \
  pip install torch-geometric

pip install -r requirements.txt

echo ""
echo "Environment ready. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Prefetch leaderboard splits:"
echo "  python download_datasets.py --prefetch_jarvis"
echo ""
echo "Example CTGP run on GPU 0:"
echo "  python ctgp.py --device cuda:0 --leaderboard --class_target metal_opt --repr_level 0 \\"
echo "    --run_ctgp --generations 300 --runs 1 --output_dir results/smoke"
