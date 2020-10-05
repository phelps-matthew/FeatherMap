import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from feathermap.models.feathernet import FeatherNet
from feathermap.models.ffnn import FFNN
from feathermap.utils import timed, set_logger
import logging
import argparse
import os


def load_data(batch_size, **kwargs):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=transforms.ToTensor()
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
def train(model, train_loader, epochs, lr, device, verbose=False):
    model.train()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Monitor for GPU
            if verbose:
                print("Outside: input size", images.size(), "output_size", outputs.size())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logging.info(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_step, loss.item()
                    )
                )


@timed
def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logging.info(
            "Accuracy of the network on the 10000 test images: {} %".format(accuracy)
        )
        return accuracy


@timed
def main(args):
    # Initialize logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(args.log_dir + "ffnn_compress_" + str(args.compress) + ".log")

    # MNIST-parameters
    input_size = 784
    num_classes = 10

    # Select model
    base_model = FFNN(input_size, args.hidden_size, num_classes)
    if args.compress:
        model = FeatherNet(base_model, compress=args.compress, constrain=args.constrain, verbose=args.verbose)
    else:
        model = base_model

    # Enable GPU support
    if torch.cuda.is_available():
        print("Utilizing", torch.cuda.device_count(), "GPU(s)!")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        DEV = torch.device("cuda:0")
        cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    else:
        print("Utilizing CPU!")
        DEV = torch.device("cpu")
        cuda_kwargs = {}
    model.to(DEV)

    # Load data
    train_loader, test_loader = load_data(args.batch_size, **cuda_kwargs)

    # Train, evaluate
    train(model, train_loader, args.epochs, args.lr, DEV, verbose=args.verbose)
    evaluate(model, test_loader, DEV)

    # Save the model checkpoint
    if args.save_model:
        torch.save(
            model.state_dict(),
            args.log_dir + "ffnn_compress_" + str(args.compress) + ".ckpt",)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser( description="FFNN on MNIST with Structured Multi-Hashing compression", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
        parser.add_argument("--hidden-size", type=int, default=500, help="Hidden size")
        parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
        parser.add_argument( "--batch-size", type=int, default=100, help="Mini-batch size")
        parser.add_argument( "--lr", type=float, default=0.001, help="Learning rate at t=0")
        parser.add_argument( "--num-workers", type=int, default=1, help="Number of dataloader processing threads. Try adjusting for faster training",)
        parser.add_argument( "--compress", type=float, default=0.5, help="Compression rate. Set to zero for base model",)
        parser.add_argument( "--save-model", action="store_true", default=False, help="Save model in local directory",)
        parser.add_argument( "--data-dir", type=str, default="./data/", help="Path to store MNIST data",)
        parser.add_argument( "--log-dir", type=str, default="./logs/", help="Path to store training and evaluation logs",)
        parser.add_argument( "--constrain", action="store_true", default=False, help="Constrain to per layer caching",)
        parser.add_argument( "--verbose", action="store_true", default=False, help="Verbose messages.",)
        args = parser.parse_args()
        print(args)
        main(args)
    except KeyboardInterrupt:
        exit()
