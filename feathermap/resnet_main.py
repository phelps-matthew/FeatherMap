# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from                                    #
# https://github.com/yunjey/pytorch-tutorial and                               #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from feathermap.resnet import ResidualBlock, ResNet
from feathermap.feathernet import FeatherNet
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ResNet34 on CIFAR10 with Structured Multi-Hashing compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate at t=0")
    parser.add_argument(
        "--compress",
        type=float,
        default=0.5,
        help="Compression rate. Set to zero for base model",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save model in local directory",
    )
    args = parser.parse_args()

    print(args)
    return args


def load_data(batch_size):
    # Image preprocessing modules
    transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
        ]
    )

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=False, transform=transforms.ToTensor()
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(model, train_loader, epochs, lr, device):
    model.train()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Train the model
    total_step = len(train_loader)
    curr_lr = lr
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_step, loss.item()
                    )
                )

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


def evaluate(model, test_loader, device):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print("Accuracy of the model on the test images: {} %".format(accuracy))
        return accuracy


def main():
    args = parse_arguments()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model
    base_model = ResNet(ResidualBlock, [2, 2, 2])
    if args.compress:
        model = FeatherNet(base_model, exclude=(nn.BatchNorm2d), compress=args.compress)
    else:
        model = base_model.to(device)

    # Load data
    train_loader, test_loader = load_data(args.batch_size)

    # Train, evaluate
    train(model, train_loader, args.epochs, args.lr, device)
    evaluate(model, test_loader, device)

    # Save the model checkpoint
    if args.save_model:
        torch.save(model.state_dict(), "resnet.ckpt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
