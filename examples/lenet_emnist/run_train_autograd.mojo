"""CLI entry point for LeNet-5 EMNIST training via tape-based autograd.

Usage:
    pixi run mojo run examples/lenet_emnist/run_train_autograd.mojo \\
        --epochs 1 --batch-size 32 --lr 0.01 --max-batches 10
"""

from train_autograd import main as train_main


def main() raises:
    train_main()
