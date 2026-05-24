"""
Distributed training utilities for AWS SageMaker / GCP / torchrun.

Usage (local multi-GPU):
    torchrun --nproc_per_node=4 scripts/train_ddp.py

SageMaker sets SM_HOSTS, SM_CURRENT_HOST, SM_NUM_GPUS automatically.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def sageMaker_env() -> Tuple[int, int, int]:
    """
    Returns (rank, world_size, local_rank) from SageMaker or torchrun env.
    """
    if "SM_CURRENT_HOST" in os.environ:
        hosts = json_load_hosts(os.environ.get("SM_HOSTS", "[]"))
        rank = hosts.index(os.environ["SM_CURRENT_HOST"])
        world_size = len(hosts)
        local_rank = int(os.environ.get("SM_LOCAL_RANK", 0))
        return rank, world_size, local_rank

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def json_load_hosts(s: str):
    import json

    return json.loads(s.replace("'", '"')) if s.startswith("[") else s.split(",")


def init_distributed(backend: Optional[str] = None) -> Tuple[int, int, torch.device]:
    rank, world_size, local_rank = sageMaker_env()
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return rank, world_size, device

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def wrap_model(model: torch.nn.Module, device: torch.device, local_rank: int) -> torch.nn.Module:
    if not is_distributed() or not torch.cuda.is_available():
        return model.to(device)
    return DDP(
        model.to(device),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_initialized():
        dist.barrier()
