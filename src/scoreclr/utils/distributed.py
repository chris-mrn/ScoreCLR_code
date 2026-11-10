"""torch.distributed helpers. Everything degrades to single-process when not launched with torchrun."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed() -> torch.device:
    """Joins the process group when launched with torchrun; returns this process's device."""
    if "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=timedelta(hours=4))  # rank 0 evaluates alone while the others wait
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(metrics: dict, device) -> dict:
    """Averages a dict of floats over processes."""
    if not is_distributed():
        return metrics
    values = torch.tensor(list(metrics.values()), dtype=torch.float64, device=device)
    dist.all_reduce(values)
    return dict(zip(metrics, (values / world_size()).tolist()))


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()
