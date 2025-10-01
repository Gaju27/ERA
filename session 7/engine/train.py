from typing import List
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch, train_losses: List[float], train_acc: List[float]) -> None:
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        pred = output.float().argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc=f"Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}")
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, test_losses: List[float], test_acc: List[float]) -> None:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.float().argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    test_acc.append(accuracy)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


