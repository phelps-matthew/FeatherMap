"""
Train compressed feathermap/models on CIFAR10.
    - Progress bar inspried by https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from packaging import version
import os
import argparse
from feathermap.utils import progress_bar
from feathermap.models.resnet import ResNet34
from feathermap.feathernet import FeatherNet
from feathermap.dataloader import get_train_valid_loader, get_test_loader


def main():
    """Perform training, validation, and testing, with checkpoint loading and saving"""

    # Build Model
    print("==> Building model..")
    base_model = ResNet34()
    if args.compress:
        model = FeatherNet(
            base_model,
            compress=args.compress,
        )
    else:
        if args.lr != 0.1:
            print("Warning: Suggest setting base-model learning rate to 0.1")
        model = base_model

    # Enable GPU support
    print("==> Setting up device..")
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

    # Create dataloaders
    print("==> Preparing data..")
    train_loader, valid_loader = get_train_valid_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        **cuda_kwargs
    )
    test_loader = get_test_loader(data_dir=args.data_dir, **cuda_kwargs)

    best_acc = 0  # best validation accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    save_display = False

    # Load checkpoint
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/" + args.ckpt_name)
        model.load_state_dict(checkpoint["model"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]

    # Initialize optimizers and loss fn
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    def train(epoch: int) -> None:
        """Train on CIFAR10 per epoch"""
        # maintain backward compatibility; get_last_lr requires PyTorch >= 1.4
        last_lr = (
            scheduler.get_last_lr()[0]
            if version.parse(torch.__version__) >= version.parse("1.4")
            else scheduler.get_lr()[0]
        )
        print(
            "\nEpoch: {}  |  Compression: {:.2f}  |  lr: {:<6}".format(
                epoch, args.compress, last_lr
            )
        )
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEV), targets.to(DEV)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: {:.3f} | Acc: {:.3f}% ({}/{})".format(
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Validation
    def validate(epoch: int) -> None:
        """Validate on CIFAR10 per epoch. Save best accuracy for checkpoint storing"""
        nonlocal best_acc
        nonlocal save_display
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(DEV), targets.to(DEV)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(valid_loader),
                    "Loss: {:.3f} | Acc: {:.3f}% ({}/{})".format(
                        valid_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        # Save checkpoint.
        acc = 100.0 * correct / total
        save_display = acc > best_acc
        if acc > best_acc:
            state = {
                "model": model.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/" + args.ckpt_name)
            best_acc = acc

    # Testing
    def test(epoch: int) -> None:
        """Test on CIFAR10 per epoch."""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(DEV), targets.to(DEV)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(test_loader),
                    "Loss: {:.3f} | Acc: {:.3f}% ({}/{})".format(
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

    # Train up to 300 epochs
    # *Displays* concurent performance on validation and test set while training,
    # but strictly uses validation set to determine early stopping
    print("==> Initiate Training..")
    for epoch in range(start_epoch, 300):
        train(epoch)
        validate(epoch)
        test(epoch)
        if save_display:
            print("Saving..")
        scheduler.step()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="PyTorch CIFAR10 training with Structured Multi-Hashing compression",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--compress",
            type=float,
            default=0.5,
            help="Compression rate. Set to zero for base model",
            metavar="",
        )
        parser.add_argument(
            "--resume", "-r", action="store_true", help="Resume from checkpoint"
        )
        parser.add_argument(
            "--ckpt-name",
            type=str,
            default="ckpt.pth",
            help="Name of checkpoint",
            metavar="",
        )
        parser.add_argument(
            "--lr",
            default=0.01,
            type=float,
            help="Learning rate. Set to 0.1 for base model (uncompressed) training.",
            metavar="",
        )
        parser.add_argument(
            "--batch-size", type=int, default=128, help="Mini-batch size", metavar=""
        )
        parser.add_argument(
            "--valid-size",
            type=float,
            default=0.1,
            help="Validation set size as fraction of train",
            metavar="",
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
        args = parser.parse_args()
        main()
    except KeyboardInterrupt:
        exit()
