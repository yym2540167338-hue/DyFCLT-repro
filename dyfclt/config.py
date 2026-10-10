"""Tiny YAML config system: ``_base_`` inheritance + ``key.sub=value`` overrides."""
import copy
import os

import yaml


def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if k == "_delete_":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_delete_", False):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy({kk: vv for kk, vv in v.items() if kk != "_delete_"} if isinstance(v, dict) else v)
    return out


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for b in bases:
        merged = _merge(merged, load_config(os.path.join(os.path.dirname(path), b)))
    return _merge(merged, cfg)


def apply_overrides(cfg, opts):
    """opts: list like ['train.epochs=1', 'model.encoder.use_dfca=false']; values parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for opt in opts or []:
        key, _, val = opt.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def dump_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
