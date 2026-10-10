#!/usr/bin/env bash
# Linux + NVIDIA GPU environment for full-scale training (paper: one A100).
# Usage: bash environment/setup_server_cuda.sh [cu121|cu118|cu124]
set -euo pipefail
CUDA_TAG=${1:-cu121}
ENV_NAME=${ENV_NAME:-dyfclt}

if ! command -v conda >/dev/null; then
  echo "conda not found: install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)"; exit 1
fi
eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

# PyTorch wheel matching the driver (check `nvidia-smi` -> "CUDA Version" >= the wheel's CUDA)
pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r requirements.txt

# RT-DETR's ImageNet-pretrained PResNet50-vd (same file the RT-DETR repo downloads), ~94 MB
mkdir -p pretrained
if [ ! -f pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth ]; then
  wget -c -O pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth \
    https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth
fi
echo "sha256 (expected 641dc598...4353):"; sha256sum pretrained/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
x = torch.randn(2, 8, 80, 80, device="cuda")
print("cuFFT ok:", torch.fft.irfft2(torch.fft.rfft2(x), s=(80, 80)).sub(x).abs().max().item() < 1e-4)
EOF
python tests/test_pipeline.py
echo "environment ready: conda activate $ENV_NAME"
