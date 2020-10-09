"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import argparse
from feathermap.train.models.resnet import ResNet34
from feathermap.train.mutils import progress_bar
from feathermap.models.feathernet import FeatherNet
from feathermap.data_loader import get_test_loader
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--epochs", type=int, default=1, help="Number of epochs to evaluate over test set"
)
parser.add_argument(
    "--compress",
    type=float,
    default=0,
    help="Compression rate. Set to zero for base model",
)
parser.add_argument(
    "--constrain",
    action="store_true",
    default=False,
    help="Constrain to per layer caching",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
    help="Number of dataloader processing threads. Try adjusting for faster training",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="./data/",
    help="Path to store CIFAR10 data",
)
args = parser.parse_args()


# Build Model
print("==> Building model..")
base_model = ResNet34()
if args.compress:
    model = FeatherNet(
        base_model,
        exclude=(nn.BatchNorm2d),
        compress=args.compress,
        constrain=args.constrain,
    )
else:
    model = base_model

# Enable GPU support
print("==> Preparing device..")
if torch.cuda.is_available():
    print("Utilizing", torch.cuda.device_count(), "GPU(s)!")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    DEV = torch.device("cuda:0")
    cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    cudnn.benchmark = True
else:
    print("Utilizing CPU!")
    DEV = torch.device("cpu")
    cuda_kwargs = {}
model.to(DEV)
if args.constrain:
    model.train_stream(False)
else:
    model.eval()


# fmt: off
import ipdb,os; ipdb.set_trace(context=30)  # noqa
# fmt: on
# Create dataloaders
print("==> Preparing data..")
test_loader = get_test_loader(data_dir=args.data_dir, batch_size=100, **cuda_kwargs)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Benchmark latency
print("==> Evaluating test set..")


def test(epoch):
    start = timer()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEV), targets.to(DEV)
            outputs = model(inputs)
    end = timer()
    return end - start


dt = []
for epoch in range(start_epoch, start_epoch + args.epochs):
    dt.append(test(epoch))

# 10k images in test set; result is fps
avg = args.epochs * 10000 / sum(dt)
print("Average fps: {:.4f}".format(avg))
print("Average latency: {:.4f}".format(1 / avg))
