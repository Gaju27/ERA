import argparse
from imports import torch, optim, tqdm
from data_utils import get_transforms, get_datasets, get_loaders
from lr_schedulers import get_scheduler
from model_strategy import get_strategy, ModelContext
from visualization import save_metrics_plot, save_model_params, save_metrics_json
from torchsummary import summary
import os

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = torch.nn.functional.nll_loss(y_pred, target)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_losses.append(loss.detach().cpu())
        train_acc.append(100 * correct / processed)
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}')

def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    test_acc.append(acc)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')

def save_model_summary(model, name_version, out_dir='summaries'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f'{name_version}_summary.txt')
    with open(save_path, 'w') as f:
        f.write(str(summary(model, input_size=(1, 28, 28))))
    print(f'Model summary saved to {save_path}')

def main():
    parser = argparse.ArgumentParser(description='MNIST Model Runner')
    parser.add_argument('--model', type=str, default='1', choices=['1', '2', '3'], help='Model version to run: 1, 2, or 3')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'onecycle'], help='Learning rate scheduler type')
    parser.add_argument('--max_lr', type=float, default=0.75, help='Max LR for OneCycleLR')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    train_transforms, test_transforms = get_transforms()
    train_dataset, test_dataset = get_datasets(train_transforms, test_transforms)
    train_loader, test_loader = get_loaders(train_dataset, test_dataset, use_cuda)

    # Strategy pattern for model selection
    strategy = get_strategy(args.model)
    context = ModelContext(strategy)
    model = context.get_model(device)

    # Generate and save model summary
    name_version = f"mnist_model_{args.model}"
    save_model_summary(model, name_version)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if args.scheduler == 'onecycle':
        scheduler = get_scheduler(
            'onecycle', optimizer, args.epochs, steps_per_epoch=len(train_loader), max_lr=args.max_lr
        )
    else:
        scheduler = get_scheduler('step', optimizer, args.epochs)

    train_losses, test_losses, train_acc, test_acc = [], [], [], []

    for epoch in range(args.epochs):
        print(f'EPOCH: {epoch}')
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        scheduler.step()
        test(model, device, test_loader, test_losses, test_acc)

    name_version = f"mnist_model_{args.model}"
    save_metrics_plot(train_losses, test_losses, train_acc, test_acc, name_version)
    save_model_params(model, name_version)
    save_metrics_json(train_losses, test_losses, train_acc, test_acc, name_version)

if __name__ == '__main__':
    main()
