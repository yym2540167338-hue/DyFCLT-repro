#!/usr/bin/env bash
# Apple-silicon (MPS) environment used for the local smoke tests in this repo.
set -euo pipefail
ENV_NAME=${ENV_NAME:-dyfclt}
eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
pip install torch torchvision          # macOS wheels include MPS support
pip install -r requirements.txt
mkdir -p pretrained
[ -f pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth ] || curl -L -o pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
python -c "import torch; print('torch', torch.__version__, 'mps:', torch.backends.mps.is_available())"
python tests/test_pipeline.py
