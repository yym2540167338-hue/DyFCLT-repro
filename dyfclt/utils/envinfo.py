"""Explain why PyTorch cannot see a CUDA GPU (used by get_device and tools/check_env.py)."""
import glob
import os
import re
import shutil
import socket
import subprocess
import sys

import torch

SCHEDULERS = (("Slurm", "sbatch", "SLURM_JOB_ID"), ("PBS", "qsub", "PBS_JOBID"), ("LSF", "bsub", "LSB_JOBID"))


def nvidia_smi():
    """(driver_version, max_cuda_version, gpu_names) from nvidia-smi, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20).stdout.split("\n")
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    text = out.stdout + out.stderr
    drv = re.search(r"Driver Version:\s*([\d.]+)", text)
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return {"driver": drv.group(1) if drv else None, "max_cuda": cuda.group(1) if cuda else None,
            "gpus": [n.strip() for n in names if n.strip() and "No devices" not in n],
            "raw_head": "\n".join(text.strip().splitlines()[:4])}


def cuda_init_error():
    """The actual CUDA initialisation error message (None if CUDA works)."""
    if torch.cuda.is_available():
        return None
    try:
        torch.cuda.init()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}".strip().splitlines()[0]
    return None


def scheduler_state():
    return {name: {"in_job": var in os.environ, "job_id": os.environ.get(var)}
            for name, binary, var in SCHEDULERS if shutil.which(binary) or var in os.environ}


def device_files():
    return {f: os.access(f, os.R_OK | os.W_OK) for f in sorted(glob.glob("/dev/nvidia[0-9]*"))}


def _ver(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:2]) or (0,)


def describe():
    smi = nvidia_smi()
    lines = [f"host        : {socket.gethostname()}",
             f"python      : {sys.executable}",
             f"conda env   : {os.environ.get('CONDA_DEFAULT_ENV', '-')}",
             f"torch       : {torch.__version__} (built with CUDA {torch.version.cuda}) from {os.path.dirname(torch.__file__)}",
             f"cuda avail. : {torch.cuda.is_available()}  device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
             f"cuda error  : {cuda_init_error()}",
             f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
             f"scheduler   : {scheduler_state() or '-'}",
             f"/dev/nvidia*: {device_files() or '-'} (True = this process may open it)",
             f"nvidia-smi  : {smi if smi is None or 'error' in smi else {k: smi[k] for k in ('driver', 'max_cuda', 'gpus')}}"]
    return lines, smi


def cuda_problems():
    """Likely reasons (in Chinese, with fixes) why torch.cuda.is_available() is False."""
    if torch.cuda.is_available():
        return []
    if sys.platform == "darwin":
        return ["macOS 没有 CUDA：完整训练需要在 Linux + NVIDIA GPU 服务器上运行（本机调试可用 DEVICE=mps）"]
    _, smi = describe()
    err = (cuda_init_error() or "").lower()
    probs = []
    for name, st in scheduler_state().items():
        if not st["in_job"]:
            probs.append(f"检测到 {name} 作业调度系统，但当前进程不在作业里：你很可能在登录节点上，登录节点不能用显卡。"
                         "请申请 GPU 节点后再运行，例如交互式 `srun -p <GPU分区> --gres=gpu:1 -c 8 --mem=64G --pty bash`，"
                         "或提交作业 `sbatch -p <GPU分区> scripts/slurm_train.sbatch`（分区名用 `sinfo` 查看）")
        elif name == "Slurm" and not any(os.environ.get(v) for v in
                                          ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")):
            probs.append("当前 Slurm 作业没有申请 GPU：提交时加 `--gres=gpu:1`（有的集群写作 `--gpus=1`）")
    files = device_files()
    if files and not all(files.values()):
        probs.append("节点上有 /dev/nvidia* 设备，但当前进程无权访问：显卡没有分配给你（登录节点，或作业没申请 GPU）")
    if "unknown error" in err:
        probs.append("CUDA 初始化报 'unknown error'：驱动在，但这个进程用不了显卡。最常见的原因是上面说的没在 GPU 作业里；"
                     "如果确实已经在分配了 GPU 的节点上，就是节点驱动状态异常（如 nvidia_uvm 模块或驱动版本不一致），需要联系集群管理员")
    if "/.local/lib/" in torch.__file__:
        probs.append(f"PyTorch 来自用户目录 {os.path.dirname(torch.__file__)}（pip install --user 装的）。它会覆盖同版本 Python 的 "
                     "conda 环境里的包，容易版本错乱。建议 `export PYTHONNOUSERSITE=1`，并按 environment/setup_server_cuda.sh 建 dyfclt 环境")
    if torch.version.cuda is None:
        probs.append("安装的是 CPU 版 PyTorch（torch.version.cuda 为 None）。在 dyfclt 环境里重装 CUDA 版："
                     "pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121"
                     "（nvidia-smi 显示的 CUDA Version < 12.1 时把 cu121 换成 cu118）")
    if "dyfclt" not in sys.executable:
        probs.append(f"当前 python 不是 dyfclt 环境（{sys.executable}）。先激活：`conda activate dyfclt`，"
                     "或没有 conda 时 `source ~/venvs/dyfclt/bin/activate`（由 environment/setup_server_cuda.sh 创建）")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1", "none", "NoDevFiles"):
        probs.append(f"CUDA_VISIBLE_DEVICES={cvd!r} 把显卡全部屏蔽了。改成 export CUDA_VISIBLE_DEVICES=0 或 unset")
    if smi is None and not files and not scheduler_state():
        probs.append("找不到 nvidia-smi，也没有 /dev/nvidia* 设备：这台机器没有可见的 NVIDIA 显卡/驱动（如果在 Docker 里，需要 --gpus all）")
    elif smi and "error" not in smi and torch.version.cuda and smi.get("max_cuda") and \
            _ver(smi["max_cuda"]) < _ver(torch.version.cuda):
        probs.append(f"显卡驱动太旧：驱动 {smi['driver']} 最高支持 CUDA {smi['max_cuda']}，而 PyTorch 需要 CUDA "
                     f"{torch.version.cuda}。重装 cu118 版 PyTorch，或升级驱动")
    if not probs:
        probs.append(f"未能自动判断原因，CUDA 报错为：{cuda_init_error()}。请把 `python tools/check_env.py` 的完整输出发给我")
    return probs
