"""Training / evaluation loops (single GPU, torchrun multi-GPU, Apple MPS or CPU)."""
import datetime
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..config import dump_config
from ..data.paired_coco import build_dataset, collate_fn
from ..evaluation.cocoeval_ext import evaluate as coco_evaluate
from ..models.detector import build_criterion, build_model, build_postprocessor
from ..utils import dist as du
from .optim import ModelEMA, WarmupStepLR, build_optimizer, de_parallel


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(msg, path=None):
    if du.is_main_process():
        print(msg, flush=True)
        if path:
            with open(path, "a") as f:
                f.write(msg + "\n")


def to_device(targets, device):
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]


def build_loader(cfg, dataset, train, distributed):
    t = cfg["train"] if train else cfg.get("eval", {})
    world = du.get_world_size()
    bs = int(t.get("batch_size", 8)) // world if train else int(t.get("batch_size", 8))
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    return DataLoader(dataset, batch_size=max(bs, 1), shuffle=train and sampler is None, sampler=sampler,
                      num_workers=int(cfg.get("num_workers", 4)), collate_fn=collate_fn, drop_last=train,
                      pin_memory=torch.cuda.is_available(), persistent_workers=int(cfg.get("num_workers", 4)) > 0)


def band_boundaries(model):
    out = {}
    for name, mod in de_parallel(model).named_modules():
        if hasattr(mod, "boundaries") and getattr(mod, "num_bands", 1) > 1:
            out[name] = [round(float(x), 4) for x in mod.boundaries().detach().cpu()]
    return out


@torch.no_grad()
def run_evaluation(model, postprocessor, loader, dataset, eval_cfg, device):
    model.eval()
    max_batches = eval_cfg.get("max_batches")
    results = []
    for i, (vis, ir, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(vis.to(device), ir.to(device))
        orig = torch.stack([t["orig_size"] for t in targets]).to(device)
        for t, d in zip(targets, postprocessor(out, orig)):
            boxes = d["boxes"].cpu()
            boxes[:, 2:] -= boxes[:, :2]
            for b, s, l in zip(boxes.tolist(), d["scores"].cpu().tolist(), d["labels"].cpu().tolist()):
                results.append({"image_id": int(t["image_id"]), "category_id": dataset.label2cat[int(l)],
                                "bbox": [round(x, 2) for x in b], "score": round(s, 5)})
    gathered = du.all_gather_object(results)
    if not du.is_main_process():
        return {}, []
    # DistributedSampler pads shards with repeated images: keep each image from one rank only
    seen, merged = set(), []
    for part in gathered:
        ids = {r["image_id"] for r in part}
        merged += [r for r in part if r["image_id"] not in seen]
        seen |= ids
    results = merged
    img_ids = sorted({r["image_id"] for r in results}) if max_batches is not None else dataset.ids
    metrics = coco_evaluate(dataset.coco, results, eval_cfg["protocol"], cat_ids=dataset.cat_ids, img_ids=img_ids,
                            safit_c=eval_cfg.get("safit_c", 32.0), nwd_anchor=eval_cfg.get("nwd_anchor", "corner"))
    return metrics, results


def train_one_epoch(model, criterion, loader, optimizer, scheduler, ema, device, epoch, cfg, log_path, scaler=None):
    model.train()
    criterion.train()
    t = cfg["train"]
    clip = float(t.get("clip_max_norm", 0.1))
    print_freq = int(t.get("print_freq", 50))
    max_iters = t.get("max_iters_per_epoch")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(t.get("amp", "none")).lower())
    n = len(loader) if max_iters is None else min(len(loader), int(max_iters))
    t0 = time.time()
    meter = {}
    for i, (vis, ir, targets) in enumerate(loader):
        if i >= n:
            break
        vis, ir = vis.to(device, non_blocking=True), ir.to(device, non_blocking=True)
        targets = to_device(targets, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            outputs = model(vis, ir, targets)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            log(f"non-finite loss {loss.item()} at epoch {epoch} iter {i}: "
                f"{ {k: float(v) for k, v in loss_dict.items() if not math.isfinite(float(v))} }", log_path)
            sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        reduced = du.reduce_dict({k: v.detach() for k, v in loss_dict.items() if "_aux_" not in k and "_dn_" not in k})
        for k, v in reduced.items():
            meter[k] = meter.get(k, 0.0) + float(v)
        meter["loss"] = meter.get("loss", 0.0) + float(loss.detach())
        if (i + 1) % print_freq == 0 or i == n - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            avg = {k: round(v / (i + 1), 4) for k, v in meter.items()}
            log(f"[epoch {epoch}] it {i + 1}/{n} lr {optimizer.param_groups[-1]['lr']:.2e} "
                f"{el / (i + 1):.2f}s/it eta {datetime.timedelta(seconds=int(eta))} {avg}", log_path)
    return {k: v / max(n, 1) for k, v in meter.items()}


def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, cfg, best, weights_only=False):
    """last.pth keeps optimizer/scheduler state for --resume; best.pth stores weights only (~half the size)."""
    if not du.is_main_process():
        return
    state = {"ema": ema.state_dict() if ema else None, "epoch": epoch, "cfg": cfg, "best": best}
    if weights_only and ema is not None:
        state["model"] = None
    else:
        state["model"] = de_parallel(model).state_dict()
    if not weights_only:
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    torch.save(state, path)


def train(cfg, resume=None, device_pref="auto"):
    local_rank = du.init_distributed()
    distributed = local_rank is not None
    if distributed:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = get_device(device_pref)
    set_seed(int(cfg.get("seed", 0)) + du.get_rank())
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    if du.is_main_process():
        dump_config(cfg, os.path.join(out_dir, "config.yaml"))
    log(f"device={device} world_size={du.get_world_size()} output_dir={out_dir}", log_path)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params: {n_params / 1e6:.2f}M (trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M)", log_path)
    criterion = build_criterion(cfg)
    post = build_postprocessor(cfg)
    train_set, val_set = build_dataset(cfg, "train"), build_dataset(cfg, "val")
    log(f"train images {len(train_set)}, val images {len(val_set)}, classes {train_set.class_names}", log_path)
    train_loader = build_loader(cfg, train_set, True, distributed)
    val_loader = build_loader(cfg, val_set, False, distributed)

    optimizer = build_optimizer(model, cfg)
    t = cfg["train"]
    iters = len(train_loader) if t.get("max_iters_per_epoch") is None else min(len(train_loader), int(t["max_iters_per_epoch"]))
    scheduler = WarmupStepLR(optimizer, iters, int(t.get("warmup_iters", 0)), t.get("lr_milestones", []), t.get("lr_gamma", 0.1))
    ema = ModelEMA(model, **cfg.get("ema", {})) if cfg.get("use_ema", True) else None
    scaler = torch.amp.GradScaler("cuda") if str(t.get("amp", "none")).lower() == "fp16" and device.type == "cuda" else None

    start_epoch, best = 0, {"metric": -1.0, "epoch": -1}
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        log(f"resumed from {resume} at epoch {start_epoch}", log_path)
    else:
        scheduler.start()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(t.get("find_unused_parameters", False)))

    ev = cfg.get("eval", {})
    key = ev.get("key_metric", "AP50")
    epochs = int(t["epochs"])
    for epoch in range(start_epoch, epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        stats = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, ema, device, epoch, cfg, log_path, scaler)
        log(f"[epoch {epoch}] train done in {datetime.timedelta(seconds=int(time.time() - t0))} {stats}", log_path)
        log(f"[epoch {epoch}] band boundaries {band_boundaries(model)}", log_path)
        do_eval = (epoch + 1) % int(ev.get("interval", 1)) == 0 or epoch == epochs - 1
        metrics = {}
        if do_eval:
            eval_model = ema.module if ema is not None else de_parallel(model)
            metrics, _ = run_evaluation(eval_model, post, val_loader, val_set, ev, device)
            log(f"[epoch {epoch}] eval {json.dumps(metrics)}", log_path)
            if du.is_main_process() and metrics.get(key, -1) > best["metric"]:
                best = {"metric": metrics[key], "epoch": epoch, "key": key}
                save_checkpoint(os.path.join(out_dir, "best.pth"), model, optimizer, scheduler, ema, epoch, cfg, best,
                                weights_only=True)
        save_checkpoint(os.path.join(out_dir, "last.pth"), model, optimizer, scheduler, ema, epoch, cfg, best)
        if du.is_main_process():
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": epoch, "train": stats, "eval": metrics}) + "\n")
    log(f"done. best {best}", log_path)
    return best
