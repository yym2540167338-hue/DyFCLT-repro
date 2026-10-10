"""Losses: RT-DETR set criterion (VFL + L1 + GIoU, aux + denoising) and the focal
Tversky loss [Abraham & Khan, ISBI'19] that supervises the IBS foreground mask."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..utils.dist import get_world_size, is_dist_available_and_initialized
from .box_ops import (box_cxcywh_to_xyxy, elementwise_box_iou, elementwise_generalized_box_iou,
                      generalized_box_iou)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.use_focal_loss, self.alpha, self.gamma = use_focal_loss, alpha, gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        if sum(sizes) == 0:
            empty = torch.zeros(0, dtype=torch.int64)
            return [(empty, empty) for _ in targets]
        logits = outputs["pred_logits"].flatten(0, 1).float()
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets]).float()
        if self.use_focal_loss:
            prob = torch.sigmoid(logits)[:, tgt_ids]
            neg = (1 - self.alpha) * prob ** self.gamma * (-(1 - prob + 1e-8).log())
            pos = self.alpha * (1 - prob) ** self.gamma * (-(prob + 1e-8).log())
            cost_class = pos - neg
        else:
            cost_class = -logits.softmax(-1)[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        c = c.view(bs, num_queries, -1).cpu()
        indices = [linear_sum_assignment(m[i]) for i, m in enumerate(c.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def box_masks_from_targets(targets, size, device):
    """Binary foreground masks (B,1,h,w): every feature cell overlapping a GT box is 1."""
    h, w = size
    out = torch.zeros(len(targets), 1, h, w, device=device)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    for i, t in enumerate(targets):
        if len(t["boxes"]) == 0:
            continue
        b = box_cxcywh_to_xyxy(t["boxes"].float())
        inx = ((xs[None] + 1) / w > b[:, 0:1]) & (xs[None] / w < b[:, 2:3])
        iny = ((ys[None] + 1) / h > b[:, 1:2]) & (ys[None] / h < b[:, 3:4])
        out[i, 0] = (iny[:, :, None] & inx[:, None, :]).any(0).float()
    return out


def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=4.0 / 3.0, smooth=1.0):
    """FTL = (1 - TI)^(1/gamma), TI = (TP+s)/(TP + alpha*FN + beta*FP + s); averaged over images."""
    p = torch.sigmoid(logits.float()).flatten(1)
    t = target.flatten(1)
    tp = (p * t).sum(1)
    fn = ((1 - p) * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1 - ti).clamp(min=1e-6).pow(1.0 / gamma).mean()


class SetCriterion(nn.Module):
    def __init__(self, matcher, num_classes, weight_dict, losses=("vfl", "boxes"), alpha=0.75, gamma=2.0,
                 mask_loss_cfg=None):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = dict(weight_dict)
        self.losses = list(losses)
        self.alpha, self.gamma = alpha, gamma
        self.mask_loss_cfg = dict(mask_loss_cfg or {})

    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_vfl(self, outputs, targets, indices, num_boxes):
        idx = self._src_idx(indices)
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        ious = elementwise_box_iou(box_cxcywh_to_xyxy(src_boxes.float()), box_cxcywh_to_xyxy(target_boxes.float())).detach()
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros(src_logits.shape[:2], dtype=torch.float32, device=src_logits.device)
        target_score_o[idx] = ious
        target_score = target_score_o.unsqueeze(-1) * target
        src_logits = src_logits.float()
        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].float()
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0).float()
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        giou = elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        return {"loss_bbox": loss_bbox, "loss_giou": (1 - giou).sum() / num_boxes}

    def _get_loss(self, name, outputs, targets, indices, num_boxes):
        return {"vfl": self.loss_vfl, "boxes": self.loss_boxes}[name](outputs, targets, indices, num_boxes)

    def _weighted(self, l_dict, suffix=""):
        return {k + suffix: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict}

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        device = targets[0]["labels"].device
        out = []
        for i, t in enumerate(targets):
            n = len(t["labels"])
            if n > 0:
                gt_idx = torch.arange(n, dtype=torch.int64, device=device).tile(dn_num_group)
                out.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                out.append((empty, empty))
        return out

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        num_boxes = torch.as_tensor([float(sum(len(t["labels"]) for t in targets))], device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        main = {k: v for k, v in outputs.items() if k in ("pred_logits", "pred_boxes")}
        indices = self.matcher(main, targets)
        indices = [(i.to(device), j.to(device)) for i, j in indices]
        for name in self.losses:
            losses.update(self._weighted(self._get_loss(name, main, targets, indices, num_boxes)))
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            idx = [(a.to(device), b.to(device)) for a, b in self.matcher(aux, targets)]
            for name in self.losses:
                losses.update(self._weighted(self._get_loss(name, aux, targets, idx, num_boxes), f"_aux_{i}"))
        if "dn_aux_outputs" in outputs:
            dn_idx = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                for name in self.losses:
                    losses.update(self._weighted(self._get_loss(name, aux, targets, dn_idx, dn_num_boxes), f"_dn_{i}"))

        mask_logits = outputs.get("mask_logits") or []
        if mask_logits and "loss_mask" in self.weight_dict:
            total = 0.0
            for m in mask_logits:
                gt = box_masks_from_targets(targets, m.shape[-2:], m.device)
                total = total + focal_tversky_loss(m, gt, **self.mask_loss_cfg)
            losses["loss_mask"] = self.weight_dict["loss_mask"] * total / len(mask_logits)
        return losses
