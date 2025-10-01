import torch
import torch.optim as optim

from models.cifernet import ciferNet
from data.loader import get_loaders
from engine.train import train, test


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = get_loaders()

    model = ciferNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    epochs = 35
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        test(model, device, test_loader, test_losses, test_acc)


if __name__ == "__main__":
    main()


