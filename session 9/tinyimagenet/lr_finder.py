import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from dataloader import get_dataloaders
from model import get_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, _ = get_dataloaders(batch_size=128)

model = get_resnet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)

lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot(suggest_lr=True)  # saves the plot
lr_finder.reset()
