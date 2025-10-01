from typing import Tuple
import torch
from torch.utils.data import DataLoader
from .dataset import CIFAR10Albumentations
from .transforms import get_train_transforms, get_test_transforms


def get_loaders(data_root: str = "./data", batch_size_train: int = 128, batch_size_test: int = 128, num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    use_cuda = torch.cuda.is_available()
    train_args = dict(shuffle=True, batch_size=batch_size_train)
    test_args = dict(shuffle=False, batch_size=batch_size_test)

    if use_cuda:
        train_args.update(dict(num_workers=num_workers, pin_memory=pin_memory))
        test_args.update(dict(num_workers=num_workers, pin_memory=pin_memory))

    train_ds = CIFAR10Albumentations(root=data_root, train=True, download=True, transform=get_train_transforms())
    test_ds = CIFAR10Albumentations(root=data_root, train=False, download=True, transform=get_test_transforms())

    train_loader = DataLoader(train_ds, **train_args)
    test_loader = DataLoader(test_ds, **test_args)
    return train_loader, test_loader


