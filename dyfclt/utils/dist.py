import os

import torch
import torch.distributed as dist


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def init_distributed():
    """Initialise torch.distributed when launched with torchrun; returns local_rank or None.
    NCCL on CUDA machines, gloo (CPU) otherwise - the latter only for testing the DDP code path."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
        dist.barrier()
        return local_rank
    return None


def all_gather_object(obj):
    if not is_dist_available_and_initialized():
        return [obj]
    out = [None] * get_world_size()
    dist.all_gather_object(out, obj)
    return out


def reduce_dict(d, average=True):
    if not is_dist_available_and_initialized():
        return d
    keys = sorted(d.keys())
    values = torch.stack([d[k].detach().float() for k in keys])
    dist.all_reduce(values)
    if average:
        values /= get_world_size()
    return dict(zip(keys, values))
