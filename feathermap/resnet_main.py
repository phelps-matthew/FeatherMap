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
from feathermap.models.resnet import ResidualBlock, ResNet
from feathermap.models.feathernet import FeatherNet
from feathermap.utils import timed, print_gpu_status, set_logger
import logging
import argparse
from feathermap.data_loader import get_train_valid_loader, get_test_loader
import numpy as np


@timed
def train(model, train_loader, valid_loader, epochs, lr, device):
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
    losses = []
    steps = []
    accuracies = []
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
                logging.info( "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_step, loss.item()
                    )
                )
                losses.append(loss.item())
                steps.append(i + 1 + total_step * epoch)

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

        # Run validation
        acc = evaluate(model, valid_loader, device)
        accuracies.append(acc)
    return steps, losses, accuracies


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

        logging.info("Accuracy of the model on the {} test images: {} %".format(total, accuracy))
        return accuracy


@timed
def main(args):
    # Initialize logger
    set_logger(args.log_dir + "resnet_main_compress_" + str(args.compress) + ".log")

    # Enable GPU support
    use_cuda = torch.cuda.is_available()
    print_gpu_status()
    if use_cuda:
        DEV = torch.device("cuda:0")
        cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    else:
        DEV = torch.device("cpu")
        cuda_kwargs = {}

    # Select model
    base_model = ResNet(ResidualBlock, [2, 2, 2])
    if args.compress:
        model = FeatherNet(
            base_model, exclude=(nn.BatchNorm2d), compress=args.compress
        ).to(DEV)
    else:
        model = base_model.to(DEV)

    # Create dataloaders
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=True,
        random_seed=42,
        valid_size=args.valid_size,
        show_sample=args.show_sample,
        **cuda_kwargs
    )
    test_loader = get_test_loader(
        data_dir=args.data_dir, batch_size=args.batch_size, **cuda_kwargs
    )

    # Train, evaluate
    steps, losses, accuracies = train(model, train_loader, valid_loader, args.epochs, args.lr, DEV)
    np.savetxt(("./logs/resnet_steps_compress_" + str(args.compress) + ".csv"), steps)
    np.savetxt(("./logs/resnet_losses_compress_" + str(args.compress) + ".csv"), losses)
    np.savetxt(("./logs/resnet_accuracies_compress_" + str(args.compress) + ".csv"), accuracies)
    #evaluate(model, test_loader, DEV)

    # Save the model checkpoint
    if args.save_model:
        torch.save(
            model.state_dict(),
            args.log_dir + "resnet_compress_" + str(args.compress) + ".ckpt",
        )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="ResNet34 on CIFAR10 with Structured Multi-Hashing compression",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
        parser.add_argument(
            "--batch-size", type=int, default=100, help="Mini-batch size"
        )
        parser.add_argument(
            "--lr", type=float, default=0.001, help="Learning rate at t=0"
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=1,
            help="Number of dataloader processing threads. Try adjusting for faster training",
        )
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
        parser.add_argument(
            "--data-dir",
            type=str,
            default="./data/",
            help="Path to store CIFAR10 data",
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs/",
            help="Path to store training and evaluation logs",
        )
        parser.add_argument(
            "--valid-size",
            type=float,
            default=0.1,
            help="Validation set size as fraction of train",
        )
        parser.add_argument(
            "--show-sample",
            action="store_true",
            default=False,
            help="Plot 9x9 sample grid of dataset",
        )
        args = parser.parse_args()
        print(args)
        main(args)
    except KeyboardInterrupt:
        exit()
