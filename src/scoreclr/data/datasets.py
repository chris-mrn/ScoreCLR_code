import torch
from torch.utils.data import DataLoader
from torchvision import datasets

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "imagenet100": 100, "imagenet1k": 1000, "fake": 10}


def num_classes(data_cfg) -> int:
    return data_cfg.get("num_classes") or NUM_CLASSES[data_cfg.dataset]


def build_dataset(data_cfg, train: bool, transform):
    """CIFAR-10/100 (downloaded to data.root), `fake` (random images, for tests), or an
    ImageFolder at data.train_path / data.val_path for anything else."""
    name = data_cfg.dataset
    if name == "cifar10":
        return datasets.CIFAR10(data_cfg.root, train=train, transform=transform, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(data_cfg.root, train=train, transform=transform, download=True)
    if name == "fake":
        size = data_cfg.get("fake_size", 64)
        crop = data_cfg.crop_size
        return datasets.FakeData(
            size, (3, crop, crop), NUM_CLASSES["fake"], transform=transform, random_offset=0 if train else size
        )
    return datasets.ImageFolder(data_cfg.train_path if train else data_cfg.val_path, transform=transform)


def build_loader(dataset, batch_size: int, train: bool, num_workers: int, sampler=None) -> DataLoader:
    """`train` shuffles and drops the last incomplete batch; `sampler` (e.g. a DistributedSampler)
    replaces the shuffling."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train and num_workers > 0,  # eval loaders are rebuilt each time
    )
