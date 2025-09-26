import os
import json
from imports import plt, torch

def save_metrics_plot(train_losses, test_losses, train_acc, test_acc, name_version, out_dir='plots'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot([t.cpu().item() if hasattr(t, 'cpu') else t for t in train_losses])
    plt.title('Training Loss')
    plt.subplot(2, 2, 2)
    plt.plot(test_losses)
    plt.title('Test Loss')
    plt.subplot(2, 2, 3)
    plt.plot(train_acc)
    plt.title('Training Accuracy')
    plt.subplot(2, 2, 4)
    plt.plot(test_acc)
    plt.title('Test Accuracy')
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{name_version}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

def save_model_params(model, name_version, out_dir='outputs'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f'{name_version}_params.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model parameters saved to {save_path}')

def save_metrics_json(train_losses, test_losses, train_acc, test_acc, name_version, out_dir='outputs'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metrics = {
        'train_losses': [float(t.cpu().item()) if hasattr(t, 'cpu') else float(t) for t in train_losses],
        'test_losses': [float(t) for t in test_losses],
        'train_acc': [float(a) for a in train_acc],
        'test_acc': [float(a) for a in test_acc],
    }
    save_path = os.path.join(out_dir, f'{name_version}_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to {save_path}')

def load_model_params(model, name_version, out_dir='outputs'):
    load_path = os.path.join(out_dir, f'{name_version}_params.pth')
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    print(f'Model parameters loaded from {load_path}')
    return model

def load_metrics_json(name_version, out_dir='outputs'):
    load_path = os.path.join(out_dir, f'{name_version}_metrics.json')
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    print(f'Metrics loaded from {load_path}')
    return metrics
