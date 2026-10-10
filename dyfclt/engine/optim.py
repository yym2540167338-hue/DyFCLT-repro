"""Optimiser, LR schedule and EMA (RT-DETR recipe; base LR from the paper)."""
import math
import re
from copy import deepcopy

import torch

NO_DECAY = re.compile(r"(^|\.)(encoder|decoder)\..*(bias$|norm.*weight$|band_raw_increments$)")


def build_optimizer(model, cfg):
    o = cfg["optimizer"]
    lr = float(o["lr"])
    backbone_lr = float(o.get("backbone_lr", lr * 0.1))
    wd = float(o.get("weight_decay", 1e-4))
    backbone, no_decay, default = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone.append(p)
        elif NO_DECAY.search(name):
            no_decay.append(p)
        else:
            default.append(p)
    groups = [
        {"params": backbone, "lr": backbone_lr, "weight_decay": wd, "name": "backbone"},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "no_decay"},
        {"params": default, "lr": lr, "weight_decay": wd, "name": "default"},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(o.get("betas", (0.9, 0.999))), weight_decay=wd)


class WarmupStepLR:
    """Per-iteration linear warm-up, then (optional) epoch-level step decay."""

    def __init__(self, optimizer, iters_per_epoch, warmup_iters=0, milestones=(), gamma=0.1):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.iters_per_epoch = max(1, iters_per_epoch)
        self.warmup_iters = warmup_iters
        self.milestones = list(milestones)
        self.gamma = gamma
        self.it = 0

    def factor(self, it):
        f = 1.0
        if self.warmup_iters > 0 and it < self.warmup_iters:
            f = (it + 1) / self.warmup_iters
        epoch = it // self.iters_per_epoch
        return f * self.gamma ** sum(epoch >= m for m in self.milestones)

    def step(self):
        self.it += 1
        f = self.factor(self.it)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * f

    def start(self):
        f = self.factor(self.it)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * f

    def state_dict(self):
        return {"it": self.it}

    def load_state_dict(self, s):
        self.it = s["it"]
        self.start()


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """RT-DETR ModelEMA: decay * (1 - exp(-updates / warmups))."""

    def __init__(self, model, decay=0.9999, warmups=2000):
        self.module = deepcopy(de_parallel(model)).eval()
        self.decay, self.warmups, self.updates = decay, warmups, 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        msd = de_parallel(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, s):
        self.module.load_state_dict(s["module"])
        self.updates = s["updates"]
