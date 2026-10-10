"""Train DyFCLT / baselines.

Single GPU:   python tools/train.py -c configs/droneperson/dyfclt.yaml
Multi GPU:    torchrun --nproc_per_node=4 tools/train.py -c configs/rgbt_tiny/dyfclt.yaml
Overrides:    --opts train.batch_size=16 train.amp=bf16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyfclt.config import apply_overrides, load_config  # noqa: E402
from dyfclt.engine.trainer import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--resume", help="checkpoint (last.pth) to resume from")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--output-dir")
    ap.add_argument("--opts", nargs="*", default=[], help="key.sub=value overrides")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.opts)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", os.path.join("outputs", os.path.splitext(os.path.basename(args.config))[0]))
    train(cfg, resume=args.resume, device_pref=args.device)
