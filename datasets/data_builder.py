import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed, class_name=None):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed, class_name)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True, class_name=None):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, distributed=distributed, class_name=class_name)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed, class_name=class_name)

    logger.info("build dataset done")
    return train_loader, test_loader
