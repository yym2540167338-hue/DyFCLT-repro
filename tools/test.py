"""Evaluate a checkpoint with the dataset's official protocol and save metrics + detections.

python tools/test.py -c configs/droneperson/dyfclt.yaml --ckpt outputs/droneperson_dyfclt/best.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dyfclt.config import apply_overrides, load_config  # noqa: E402
from dyfclt.data.paired_coco import build_dataset  # noqa: E402
from dyfclt.engine.trainer import band_boundaries, build_loader, get_device, run_evaluation  # noqa: E402
from dyfclt.models.detector import build_model, build_postprocessor  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="auto", help="auto (EMA if present) | ema | model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="metrics json (default: next to the checkpoint)")
    ap.add_argument("--save-dets", action="store_true", help="also dump COCO-format detections")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.opts)
    cfg["model"]["backbone"]["pretrained"] = False  # weights come from the checkpoint
    device = get_device(args.device)
    model = build_model(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    use_ema = args.weights == "ema" or (args.weights == "auto" and ck.get("ema")) or ck.get("model", 0) is None
    state = ck["ema"]["module"] if use_ema else ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {'EMA' if use_ema else 'model'} weights from {args.ckpt} (epoch {ck.get('epoch')}), "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    dataset = build_dataset(cfg, "val")
    loader = build_loader(cfg, dataset, False, False)
    metrics, dets = run_evaluation(model, build_postprocessor(cfg), loader, dataset, cfg["eval"], device)
    metrics["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    metrics["band_boundaries"] = band_boundaries(model)
    print(json.dumps(metrics, indent=1))
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "test_metrics.json")
    json.dump({"config": args.config, "ckpt": args.ckpt, "protocol": cfg["eval"]["protocol"], "metrics": metrics},
              open(out, "w"), indent=1)
    print("saved", out)
    if args.save_dets:
        p = os.path.splitext(out)[0] + "_dets.json"
        json.dump(dets, open(p, "w"))
        print("saved", p)
