from imports import torch, datasets, transforms

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transforms, test_transforms

def get_datasets(train_transforms, test_transforms, data_path='./data'):
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=test_transforms)
    return train_dataset, test_dataset

def get_loaders(train_dataset, test_dataset, use_cuda):
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader
