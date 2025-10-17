import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders
from model import get_resnet50
from train import train_one_epoch
from test import validate
from utils import save_checkpoint, plot_metrics
from visualize import compute_confusion_matrix
from gradcam import show_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders()
model = get_resnet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

num_epochs = 20
metrics = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    metrics["train_loss"].append(train_loss)
    metrics["train_acc"].append(train_acc)
    metrics["val_loss"].append(val_loss)
    metrics["val_acc"].append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    save_checkpoint({"epoch": epoch+1,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()},
                    filename=f"checkpoint_epoch{epoch+1}.pth")

# Plot metrics
plot_metrics(metrics)

# Confusion matrix & per-class accuracy
class_names = [cls[0] for cls in train_loader.dataset.classes]
compute_confusion_matrix(model, val_loader, device, class_names)

# Grad-CAM visualization for first 5 validation images
for i in range(5):
    show_gradcam(val_loader.dataset[i][0].to(device), model, model.layer4[-1])
