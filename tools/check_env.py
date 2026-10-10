"""Check that PyTorch can use the GPU before starting a long run.

python tools/check_env.py        (exit code 1 if no CUDA GPU is usable)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dyfclt.utils.envinfo import cuda_problems, describe  # noqa: E402

if __name__ == "__main__":
    lines, smi = describe()
    print("\n".join(lines))
    if smi and "raw_head" in smi:
        print(smi["raw_head"])
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(2, 8, 80, 80, device="cuda")
        err = (torch.fft.irfft2(torch.fft.rfft2(x), s=(80, 80)) - x).abs().max().item()
        print(f"CUDA tensor + cuFFT test ok (max err {err:.1e}) -> training will run on the GPU")
        sys.exit(0)
    print("\nPyTorch 看不到 CUDA 显卡，训练会失败（或在 CPU 上极慢地运行）。可能原因：")
    for i, p in enumerate(cuda_problems(), 1):
        print(f"  {i}. {p}")
    sys.exit(1)
