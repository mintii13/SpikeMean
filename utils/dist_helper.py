import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    Modified to support single GPU training
    """
    num_gpus = torch.cuda.device_count()
    
    # Check if CUDA is available
    if num_gpus == 0:
        print("No CUDA GPUs detected. Running on CPU.")
        return 0, 1

    if "SLURM_JOB_ID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torch.distributed.launch environment
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
    else:
        # Single GPU/CPU training - no distributed environment detected
        print("No distributed environment detected. Running on single GPU.")
        rank = 0
        world_size = 1
        
        # Set environment variables for compatibility
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        else:
            os.environ["MASTER_PORT"] = "29500"

    # Set CUDA device
    if num_gpus > 0:
        torch.cuda.set_device(rank % num_gpus)
        print(f"Using GPU {rank % num_gpus}")

    # Only initialize process group if world_size > 1
    if world_size > 1:
        print(f"Initializing distributed training: rank {rank}, world_size {world_size}")
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
    else:
        print("Single device training - skipping distributed initialization")
        
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    """Get current process rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    """Get world size"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_main_process():
    """Check if current process is main process"""
    return get_rank() == 0