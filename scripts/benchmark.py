#!/usr/bin/env python3
"""Full benchmark: FP32 vs mixed precision over multiple runs."""
import argparse
import json
import time
import numpy as np
import jax
from mixed_precision.data.cifar100 import load_cifar100, make_batches
from mixed_precision.models.resnet import ResNet
from mixed_precision.models.resnet_mixed import MixedPrecisionResNet
from mixed_precision.training.trainer_fp32 import (
    create_train_state,
    train_step_fp32,
    eval_step,
)
from mixed_precision.training.trainer_mixed import (
    create_mixed_train_state,
    train_step_mixed,
    eval_step_mixed,
)
from mixed_precision.training.loss_scaling import DynamicLossScaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-runs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", default="results/benchmark_results.json")
    args = p.parse_args()

    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "lr": 0.1,
        "weight_decay": 5e-4,
    }

    print("Loading CIFAR-100...")
    (train_images, train_labels), (test_images, test_labels) = load_cifar100()

    fp32_results = {
        "test_accs": [],
        "train_times": [],
        "peak_memory": [],
        "epoch_times": [],
    }
    mixed_results = {
        "test_accs": [],
        "train_times": [],
        "peak_memory": [],
        "epoch_times": [],
    }

    for run in range(args.num_runs):
        seed = run
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.num_runs} (seed={seed})")
        print(f"{'='*60}")

        # FP32
        print("  Training FP32 baseline...")
        rng = jax.random.PRNGKey(seed)
        model_fp32 = ResNet(num_classes=100)
        state_fp32 = create_train_state(
            rng, model_fp32, config["lr"], config["weight_decay"]
        )
        jax.clear_caches()
        t0 = time.time()
        epoch_times_fp32 = []
        for epoch in range(config["num_epochs"]):
            t_epoch = time.time()
            rng, epoch_rng = jax.random.split(rng)
            for batch_imgs, batch_lbls in make_batches(
                train_images, train_labels, config["batch_size"], epoch_rng
            ):
                state_fp32, loss, acc = train_step_fp32(
                    state_fp32, batch_imgs, batch_lbls
                )
            jax.block_until_ready(state_fp32.params)
            epoch_times_fp32.append(time.time() - t_epoch)
        total_time_fp32 = time.time() - t0
        test_acc_fp32 = float(eval_step(state_fp32, test_images, test_labels))
        try:
            mem_fp32 = (
                jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
            )
        except Exception:
            mem_fp32 = 0.0
        fp32_results["test_accs"].append(test_acc_fp32)
        fp32_results["train_times"].append(total_time_fp32)
        fp32_results["peak_memory"].append(mem_fp32)
        fp32_results["epoch_times"].append(np.mean(epoch_times_fp32))

        # Mixed
        print("  Training mixed precision...")
        rng = jax.random.PRNGKey(seed)
        model_mixed = MixedPrecisionResNet(num_classes=100)
        state_mixed = create_mixed_train_state(
            rng, model_mixed, config["lr"], config["weight_decay"]
        )
        loss_scaler = DynamicLossScaler()
        jax.clear_caches()
        t0 = time.time()
        epoch_times_mixed = []
        for epoch in range(config["num_epochs"]):
            t_epoch = time.time()
            rng, epoch_rng = jax.random.split(rng)
            for batch_imgs, batch_lbls in make_batches(
                train_images, train_labels, config["batch_size"], epoch_rng
            ):
                state_mixed, loss, acc = train_step_mixed(
                    state_mixed, batch_imgs, batch_lbls, loss_scaler
                )
            jax.block_until_ready(state_mixed.master_params)
            epoch_times_mixed.append(time.time() - t_epoch)
        total_time_mixed = time.time() - t0
        test_acc_mixed = float(
            eval_step_mixed(state_mixed, test_images, test_labels)
        )
        try:
            mem_mixed = (
                jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
            )
        except Exception:
            mem_mixed = 0.0
        mixed_results["test_accs"].append(test_acc_mixed)
        mixed_results["train_times"].append(total_time_mixed)
        mixed_results["peak_memory"].append(mem_mixed)
        mixed_results["epoch_times"].append(np.mean(epoch_times_mixed))

        print(
            f"  FP32:  acc={test_acc_fp32:.4f}, time={total_time_fp32:.1f}s, mem={mem_fp32:.2f}GB"
        )
        print(
            f"  Mixed: acc={test_acc_mixed:.4f}, time={total_time_mixed:.1f}s, mem={mem_mixed:.2f}GB"
        )

    # Summary (strings for display) and raw arrays for plotting
    fp32_accs = fp32_results["test_accs"]
    mixed_accs = mixed_results["test_accs"]
    summary = {
        "fp32": {
            "test_acc": f"{np.mean(fp32_accs):.4f} ± {np.std(fp32_accs):.4f}",
            "avg_train_time": f"{np.mean(fp32_results['train_times']):.1f}s",
            "avg_epoch_time": f"{np.mean(fp32_results['epoch_times']):.3f}s",
            "peak_memory_gb": f"{np.mean(fp32_results['peak_memory']):.2f} GB",
        },
        "mixed": {
            "test_acc": f"{np.mean(mixed_accs):.4f} ± {np.std(mixed_accs):.4f}",
            "avg_train_time": f"{np.mean(mixed_results['train_times']):.1f}s",
            "avg_epoch_time": f"{np.mean(mixed_results['epoch_times']):.3f}s",
            "peak_memory_gb": f"{np.mean(mixed_results['peak_memory']):.2f} GB",
        },
        "deltas": {
            "accuracy_delta": f"{np.mean(fp32_accs) - np.mean(mixed_accs):.4f}",
            "memory_reduction": f"{(1 - np.mean(mixed_results['peak_memory']) / max(np.mean(fp32_results['peak_memory']), 1e-9)) * 100:.1f}%",
            "speedup": f"{(1 - np.mean(mixed_results['train_times']) / np.mean(fp32_results['train_times'])) * 100:.1f}%",
        },
    }

    out = {
        "summary": summary,
        "fp32": fp32_results,
        "mixed": mixed_results,
    }

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({args.num_runs} runs)")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
