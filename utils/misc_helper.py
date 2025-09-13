import importlib
import logging
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def save_checkpoint(state, is_best, config):
    folder = config.save_path

    torch.save(state, os.path.join(folder, "ckpt.pth.tar"))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, "ckpt.pth.tar"),
            os.path.join(folder, "ckpt_best.pth.tar"),
        )

    if config.saver.get(
        "always_save", True
    ):  # default: save checkpoint after validate()
        epoch = state["epoch"]
        shutil.copyfile(
            os.path.join(folder, "ckpt.pth.tar"),
            os.path.join(folder, f"ckpt_{epoch}.pth.tar"),
        )


def load_state(path, model, optimizer=None):

    rank = 0#dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func, weights_only=False)

        # fix size mismatch error
        ignore_keys = []
        for k, v in checkpoint["state_dict"].items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint["state_dict"].pop(k)

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if rank == 0:
            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_metric"]
            epoch = checkpoint["epoch"]
            # optimizer.load_state_dict(checkpoint["optimizer"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (Epoch {})".format(
                        path, epoch
                    )
                )
            return best_metric, epoch
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if "image" in input:
        input["image"] = input["image"].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}


def update_config(config):
    # update feature size
    _, reconstruction_type = config.net[2].type.rsplit(".", 1)
    if reconstruction_type == "UniAD":
        input_size = config.dataset.input_size
        outstride = config.net[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]
        config.net[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.net[0].kwargs.pop("outlayers")
        config.net[0].kwargs.outblocks = outblocks
    config.net[0].kwargs.outstrides = outstrides
    config.net[1].kwargs.outplanes = [sum(outplanes)]

    return config


def init_wandb(config, args=None):
    """Initialize wandb with configuration from config file"""
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not available. Install with 'pip install wandb'")
        return None
    
    wandb_config = config.get("wandb", None)
    if not wandb_config or not wandb_config.get("enabled", False):
        return None
    
    # Setup wandb configuration
    wandb_init_kwargs = {
        "project": wandb_config.get("project", "uniad"),
        "name": wandb_config.get("name", None),
        "tags": wandb_config.get("tags", []),
        "notes": wandb_config.get("notes", ""),
        "config": dict(config),  # Log the entire config
    }
    
    # Handle entity (team/username)
    if wandb_config.get("entity"):
        wandb_init_kwargs["entity"] = wandb_config["entity"]
    
    # Handle run group and job type
    if wandb_config.get("group"):
        wandb_init_kwargs["group"] = wandb_config["group"]
    if wandb_config.get("job_type"):
        wandb_init_kwargs["job_type"] = wandb_config["job_type"]
    
    # Handle resume functionality
    if wandb_config.get("resume"):
        wandb_init_kwargs["resume"] = wandb_config["resume"]
        if wandb_config.get("run_id"):
            wandb_init_kwargs["id"] = wandb_config["run_id"]
    
    # Set wandb mode (online, offline, disabled)
    mode = wandb_config.get("mode", "online")
    wandb_init_kwargs["mode"] = mode
    
    # Handle wandb directory
    if wandb_config.get("dir"):
        wandb_init_kwargs["dir"] = wandb_config["dir"]
    
    # Login if API key provided
    if wandb_config.get("api_key"):
        wandb.login(key=wandb_config["api_key"])
    elif wandb_config.get("login", True):  # Default to True for backwards compatibility
        try:
            wandb.login()
        except Exception as e:
            print(f"Warning: wandb login failed: {e}")
            return None
    
    try:
        run = wandb.init(**wandb_init_kwargs)
        
        # Log additional metadata
        if args:
            run.config.update({"args": vars(args)})
        
        print(f"Initialized wandb run: {run.name} ({run.id})")
        return run
    
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None
