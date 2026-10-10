#!/usr/bin/env bash
# Linux + NVIDIA GPU environment for full-scale training (paper: one A100).
# Usage: bash environment/setup_server_cuda.sh [cu121|cu118|cu124]
#   * with conda:    creates conda env "dyfclt" (Python 3.11)
#   * without conda: creates a venv at ~/venvs/dyfclt from an existing Python 3.8-3.12
#                    (choose the interpreter with PYTHON=/path/to/python3.x)
# On a cluster, run this on the login node (it needs internet); the GPU check at the end only
# passes on a GPU node - repeat `python tools/check_env.py` there (e.g. inside srun/sbatch).
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_TAG=${1:-cu121}
ENV_NAME=${ENV_NAME:-dyfclt}
VENV_DIR=${VENV_DIR:-$HOME/venvs/$ENV_NAME}
export PYTHONNOUSERSITE=1 PIP_USER=0   # ignore / never install into ~/.local (pip install --user)

if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda create -y -n "$ENV_NAME" python=3.11
  conda activate "$ENV_NAME"
  ACTIVATE="conda activate $ENV_NAME"
else
  PY=""
  for cand in ${PYTHON:-} python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
    if command -v "$cand" >/dev/null && "$cand" -c \
        'import sys, venv, ensurepip; sys.exit(not (3, 8) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
      PY=$(command -v "$cand"); break
    fi
  done
  if [ -z "$PY" ]; then
    echo "conda not found, and no Python 3.8-3.12 with venv support either."
    echo "Try: module avail 2>&1 | grep -i -E 'conda|python'  ->  module load <that module>, then rerun this script."
    exit 1
  fi
  echo "conda not found -> creating a venv at $VENV_DIR with $("$PY" --version 2>&1) ($PY)"
  "$PY" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  ACTIVATE="source $VENV_DIR/bin/activate"
fi
python -m pip install --upgrade pip

# PyTorch wheel matching the driver (check `nvidia-smi` on a GPU node -> "CUDA Version" >= the wheel's CUDA)
pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r requirements.txt

# RT-DETR's ImageNet-pretrained PResNet50-vd (same file the RT-DETR repo downloads), ~94 MB
W=pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
mkdir -p pretrained
if [ ! -f "$W" ]; then
  wget -c -O "$W" https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
fi
echo "sha256 (expected 641dc598...4353):"; sha256sum "$W" 2>/dev/null || shasum -a 256 "$W"

python tests/test_pipeline.py
python tools/check_env.py || echo ">>> 当前节点用不了 GPU（登录节点通常如此）。请在 GPU 节点上再运行一次 python tools/check_env.py"
echo "environment ready. Activate it with:  $ACTIVATE && export PYTHONNOUSERSITE=1"
