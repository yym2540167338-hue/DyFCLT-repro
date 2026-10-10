"""Full DyFCLT detector: two ResNet-50 backbones -> DyFCLT encoder -> RT-DETR decoder."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .box_ops import box_cxcywh_to_xyxy
from .criterion import HungarianMatcher, SetCriterion
from .decoder import RTDETRDecoder
from .encoder import DyFCLTEncoder


class DyFCLT(nn.Module):
    def __init__(self, backbone_vis, backbone_ir, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = list(multi_scale) if multi_scale else None

    def forward(self, img_vis, img_ir, targets=None):
        if self.multi_scale and self.training:
            sz = random.choice(self.multi_scale)  # RT-DETR v1 multi-scale training ("random resize")
            img_vis = F.interpolate(img_vis, size=[sz, sz])
            img_ir = F.interpolate(img_ir, size=[sz, sz])
        feats, mask_logits = self.encoder(self.backbone_ir(img_ir), self.backbone_vis(img_vis))
        out = self.decoder(feats, targets)
        if self.training:
            out["mask_logits"] = mask_logits
        return out


class PostProcessor(nn.Module):
    """RT-DETR post-processing: top-k over (query, class) scores, boxes rescaled to the original size."""

    def __init__(self, num_classes, num_top_queries=300):
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries

    @torch.no_grad()
    def forward(self, outputs, orig_sizes):
        logits, boxes = outputs["pred_logits"].float(), outputs["pred_boxes"].float()
        bbox = box_cxcywh_to_xyxy(boxes) * orig_sizes.repeat(1, 2).unsqueeze(1).float()
        scores = torch.sigmoid(logits)
        k = min(self.num_top_queries, scores.shape[1] * scores.shape[2])
        scores, index = torch.topk(scores.flatten(1), k, dim=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        bbox = bbox.gather(1, index.unsqueeze(-1).repeat(1, 1, 4))
        return [{"labels": l, "boxes": b, "scores": s} for l, b, s in zip(labels, bbox, scores)]


def build_model(cfg):
    m = cfg["model"]
    bb_cfg = dict(m["backbone"])
    backbone_vis = build_backbone(bb_cfg)
    backbone_ir = build_backbone(bb_cfg)
    enc_cfg = dict(m["encoder"])
    enc_cfg.setdefault("in_channels", backbone_ir.out_channels)
    encoder = DyFCLTEncoder(**enc_cfg)
    dec_cfg = dict(m["decoder"])
    dec_cfg["num_classes"] = cfg["num_classes"]
    dec_cfg.setdefault("feat_channels", encoder.out_channels)
    decoder = RTDETRDecoder(**dec_cfg)
    return DyFCLT(backbone_vis, backbone_ir, encoder, decoder, multi_scale=m.get("multi_scale"))


def build_criterion(cfg):
    c = cfg["criterion"]
    matcher = HungarianMatcher(**c["matcher"])
    return SetCriterion(matcher, cfg["num_classes"], c["weight_dict"], losses=c.get("losses", ("vfl", "boxes")),
                        alpha=c.get("alpha", 0.75), gamma=c.get("gamma", 2.0), mask_loss_cfg=c.get("mask_loss"))


def build_postprocessor(cfg):
    return PostProcessor(cfg["num_classes"], cfg.get("postprocess", {}).get("num_top_queries", 300))
