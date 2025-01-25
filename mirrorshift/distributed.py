"""
This file contains a bunch of functions to support distributed on multi-nvidia-gpu setups.
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup_distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def fsdp_wrap(model: torch.nn.Module) -> FSDP:
    return FSDP(
        model,
        device_id=torch.cuda.current_device()
    )

def cleanup_distributed() -> None:
    dist.destroy_process_group()
