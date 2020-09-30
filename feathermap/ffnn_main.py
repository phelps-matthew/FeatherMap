import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from feathermap.models.feathernet import FeatherNet
from feathermap.models.ffnn import FFNN, parse_arguments
from feathermap.utils import timed, print_gpu_status, set_logger
import logging


def load_data(batch_size, **kwargs):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
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
                logging.info(
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
        logging.info("Accuracy of the network on the 10000 test images: {} %".format(accuracy))
        return accuracy


@timed
def main():
    args = parse_arguments()

    # Initialize logger
    set_logger("logs/ffnn_main_compress_" + str(args.compress) + ".log")

    # Enable GPU support
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_gpu_status()
    # Device configuration
    kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )

    # MNIST-parameters
    input_size = 784
    num_classes = 10

    # Select model
    base_model = FFNN(input_size, args.hidden_size, num_classes)
    if args.compress:
        model = FeatherNet(base_model, compress=args.compress).to(DEV)
    else:
        model = base_model.to(DEV)

    # Load data
    train_loader, test_loader = load_data(args.batch_size, **kwargs)

    # Train, evaluate
    # train(model, train_loader, args.epochs, args.lr, DEV)

    @timed
    def long_eval(num, model, test_loader, DEV):
        for i in range(num):
            evaluate(model, test_loader, DEV)

    long_eval(100, model, test_loader, DEV)

    # Save the model checkpoint
    #torch.save(model.state_dict(), "logs/ffnn_compress_" + str(args.compress) + ".ckpt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
