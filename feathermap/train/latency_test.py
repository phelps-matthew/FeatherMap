"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from feathermap.train.models.resnet import ResNet34
from feathermap.models.feathernet import FeatherNet
from feathermap.data_loader import get_test_loader
from timeit import default_timer as timer

parser = argparse.ArgumentParser(
    description="PyTorch CIFAR10 Training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    help="Number of epochs to evaluate over test set",
    metavar="",
)
parser.add_argument(
    "--compress",
    type=float,
    default=0,
    help="Compression rate. Set to zero for base model",
    metavar="",
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
    metavar="",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="./data/",
    help="Path to store CIFAR10 data",
    metavar="",
)
parser.add_argument(
    "--pin-memory",
    type=bool,
    default=False,
    help="Pin GPU memory",
    metavar="",
)
parser.add_argument(
    "--cudabench",
    type=bool,
    default=False,
    help="Set cudann.benchmark to true for static model and input",
    metavar="",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    default=False,
    help="Use CPU",
)
parser.add_argument(
    "--deploy",
    action="store_true",
    default=False,
    help="Calculate weights on the fly in eval mode",
)
parser.add_argument(
    "--v",
    action="store_true",
    default=False,
    help="Verbose",
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
        verbose=args.v
    )
else:
    model = base_model

# Enable GPU support
print("==> Preparing device..")
if torch.cuda.is_available() and not args.cpu:
    print("Utilizing", torch.cuda.device_count(), "GPU(s)!")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    DEV = torch.device("cuda:0")
    cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": args.pin_memory}
    cudnn.benchmark = args.cudabench
else:
    print("Utilizing CPU!")
    DEV = torch.device("cpu")
    cuda_kwargs = {}
model.to(DEV)
if args.deploy:
    model.deploy()
else:
    model.eval()

print(cuda_kwargs)
print("cudabench: {}".format(cudnn.benchmark))
# Create dataloaders
print("==> Preparing data..")
test_loader = get_test_loader(data_dir=args.data_dir, batch_size=100, **cuda_kwargs)

# Benchmark latency
print("==> Evaluating test set..")


def test(epoch):
    start = timer()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx == 3:
                break
            inputs, targets = inputs.to(DEV), targets.to(DEV)
            # fmt: off
            #import ipdb,os; ipdb.set_trace(context=30)  # noqa
            # fmt: on
            outputs = model(inputs)
            print(outputs)
    end = timer()
    return end - start


dt = []
for epoch in range(0, args.epochs):
    dt.append(test(epoch))

# 10k images in test set; result is fps
avg = args.epochs * 10000 / sum(dt)
print("Average fps: {:.4f}".format(avg))
print("Average latency: {:.4f}".format(1 / avg))
