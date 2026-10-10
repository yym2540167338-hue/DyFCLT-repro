"""Parameter count per component (paper Table 1 reports 85.5M for DyFCLT on RGBT-Tiny).

python tools/count_params.py configs/rgbt_tiny/dyfclt.yaml configs/rgbt_tiny/baseline.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyfclt.config import load_config  # noqa: E402
from dyfclt.models.detector import build_model  # noqa: E402


def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


if __name__ == "__main__":
    for path in sys.argv[1:] or ["configs/rgbt_tiny/dyfclt.yaml"]:
        cfg = load_config(path)
        cfg["model"]["backbone"]["pretrained"] = False
        model = build_model(cfg)
        enc = model.encoder
        parts = {
            "backbones (2x ResNet-50)": count(model.backbone_vis) + count(model.backbone_ir),
            "encoder: input proj + AIFI": sum(count(b.input_proj) + count(b.aifi) for b in (enc.ir, enc.vis)),
            "encoder: DFCA": sum(count(b.dfca) for b in (enc.ir, enc.vis) if b.dfca is not None),
            "encoder: SSE / top-down": sum(count(b.sse) for b in (enc.ir, enc.vis)),
            "encoder: PAN": sum(count(b.downsample_convs) + count(b.pan_blocks) for b in (enc.ir, enc.vis) if b.use_pan),
            "encoder: cross-modal fusion": count(enc.fusion),
            "decoder (RT-DETR)": count(model.decoder),
        }
        print(f"== {path}: total {count(model):.2f}M (paper DyFCLT: 85.5M)")
        for k, v in parts.items():
            print(f"   {k:32s} {v:7.2f}M")
