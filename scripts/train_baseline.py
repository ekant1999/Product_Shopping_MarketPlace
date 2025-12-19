#!/usr/bin/env python3
"""Train FP32 baseline ResNet on CIFAR-100."""
import argparse
import time
from mixed_precision.data.cifar100 import load_cifar100, make_batches
from mixed_precision.models.resnet import ResNet
from mixed_precision.training.trainer_fp32 import (
    create_train_state,
    train_step_fp32,
    eval_step,
    train_fp32,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    args = p.parse_args()

    config = {
        "seed": args.seed,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    print("Loading CIFAR-100...")
    train_data, test_data = load_cifar100(batch_size=config["batch_size"])
    model = ResNet(num_classes=100)

    print(f"Training FP32 baseline: epochs={config['num_epochs']}, lr={config['lr']}, batch_size={config['batch_size']}")
    t0 = time.perf_counter()
    state, test_acc = train_fp32(model, train_data, test_data, config)
    elapsed = time.perf_counter() - t0
    print(f"Done. Test accuracy: {test_acc:.4f}, time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
