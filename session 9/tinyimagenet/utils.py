import torch
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def plot_metrics(metrics_dict, save_path="metrics.png"):
    plt.figure(figsize=(10,5))
    plt.plot(metrics_dict["train_loss"], label="Train Loss")
    plt.plot(metrics_dict["val_loss"], label="Val Loss")
    plt.plot(metrics_dict["train_acc"], label="Train Acc")
    plt.plot(metrics_dict["val_acc"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
