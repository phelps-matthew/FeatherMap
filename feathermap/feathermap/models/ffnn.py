"""Fully connected feed forward neural network with one hidden layer"""
from torch import nn
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
    args = parser.parse_args()

    print(args)
    return args


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
