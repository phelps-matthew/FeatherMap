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
from feathermap.models.resnet import ResidualBlock, ResNet, parse_arguments
from feathermap.models.feathernet import FeatherNet
from feathermap.utils import timed, print_gpu_status


def load_data(batch_size, **kwargs):
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
        root="./data/", train=True, transform=transform, download=True, **kwargs
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/", train=False, transform=transforms.ToTensor()
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return train_loader, test_loader


@timed
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


@timed
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


@timed
def main():
    args = parse_arguments()

    # Enable GPU support
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_gpu_status()
    # Device configuration
    kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )

    # Select model
    base_model = ResNet(ResidualBlock, [2, 2, 2])
    if args.compress:
        model = FeatherNet(base_model, exclude=(nn.BatchNorm2d), compress=args.compress)
    else:
        model = base_model.to(DEV)

    # Load data
    train_loader, test_loader = load_data(args.batch_size, **kwargs)

    # Train, evaluate
    train(model, train_loader, args.epochs, args.lr, DEV)
    evaluate(model, test_loader, DEV)

    # Save the model checkpoint
    if args.save_model:
        torch.save(model.state_dict(), "ffnn_compress_" + str(args.compress) + ".ckpt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
