 from torch.utils.data import DataLoader
 from torchvision import datasets
 from .transforms import get_transforms


 def get_data_loaders(cfg: dict):
     train_tf = get_transforms(cfg, is_training=True)
     test_tf = get_transforms(cfg, is_training=False)

     train_ds = datasets.CIFAR10(root=cfg["data_dir"], train=True, download=True, transform=train_tf)
     test_ds = datasets.CIFAR10(root=cfg["data_dir"], train=False, download=True, transform=test_tf)

     train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
     val_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
     test_loader = val_loader

     return train_loader, val_loader, test_loader


