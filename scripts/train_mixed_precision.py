#!/usr/bin/env python3
"""Train mixed-precision ResNet on CIFAR-100."""
import argparse
import time
from mixed_precision.data.cifar100 import load_cifar100, make_batches
from mixed_precision.models.resnet_mixed import MixedPrecisionResNet
from mixed_precision.training.trainer_mixed import (
    create_mixed_train_state,
    train_step_mixed,
    eval_step_mixed,
)
from mixed_precision.training.loss_scaling import DynamicLossScaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--loss-scaling", choices=["static", "dynamic"], default="dynamic")
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
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    import jax
    rng = jax.random.PRNGKey(config["seed"])
    rng, init_rng = jax.random.split(rng)
    model = MixedPrecisionResNet(num_classes=100)
    state = create_mixed_train_state(
        init_rng, model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_scaler = DynamicLossScaler()

    print(f"Training mixed precision: epochs={config['num_epochs']}, lr={config['lr']}")
    t0 = time.perf_counter()
    for epoch in range(config["num_epochs"]):
        rng, epoch_rng = jax.random.split(rng)
        for batch_imgs, batch_lbls in make_batches(
            train_images, train_labels, config["batch_size"], epoch_rng
        ):
            state, loss, acc = train_step_mixed(
                state, batch_imgs, batch_lbls, loss_scaler
            )
        jax.block_until_ready(state.master_params)
    test_acc = float(eval_step_mixed(state, test_images, test_labels))
    elapsed = time.perf_counter() - t0
    print(f"Done. Test accuracy: {test_acc:.4f}, time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
