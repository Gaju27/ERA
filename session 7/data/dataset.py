from typing import Optional, Callable, Tuple
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


class CIFAR10Albumentations(Dataset):
    def __init__(self, root: str, train: bool = True, download: bool = False, transform: Optional[Callable] = None) -> None:
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        image, label = self.cifar10[index]
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
        return image, label

    def __len__(self) -> int:
        return len(self.cifar10)


