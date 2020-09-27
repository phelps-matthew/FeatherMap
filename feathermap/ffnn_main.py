import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from feathermap.feathernet import FeatherNet
from feathermap.ffnn import FFNN
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FFNN on MNIST with Structured Multi-Hashing compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hidden-size", type=int, default=500, help="Hidden size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
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
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
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

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_step, loss.item()
                    )
                )


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
        print("Accuracy of the network on the 10000 test images: {} %".format(accuracy))
        return accuracy


def main():
    args = parse_arguments()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST-parameters
    input_size = 784
    num_classes = 10

    # Select model
    base_model = FFNN(input_size, args.hidden_size, num_classes)
    if args.compress:
        model = FeatherNet(base_model, compress=args.compress).to(device)
    else:
        model = base_model.to(device)

    # Load data
    train_loader, test_loader = load_data(args.batch_size)

    # Train, evaluate
    train(model, train_loader, args.epochs, args.lr, device)
    evaluate(model, test_loader, device)

    # Save the model checkpoint
    if args.save_model:
        torch.save(model.state_dict(), "ffnn.ckpt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
