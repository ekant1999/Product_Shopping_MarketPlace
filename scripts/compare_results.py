#!/usr/bin/env python3
"""Generate comparison plots from benchmark_results.json."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="results/benchmark_results.json",
        help="Path to benchmark_results.json",
    )
    p.add_argument(
        "--output",
        default="results/comparison.png",
        help="Output plot path",
    )
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Support both flat format (raw arrays) and nested (summary + fp32/mixed)
    if "fp32" in data and "test_accs" in data["fp32"]:
        fp32_accs = data["fp32"]["test_accs"]
        mixed_accs = data["mixed"]["test_accs"]
        fp32_times = data["fp32"]["train_times"]
        mixed_times = data["mixed"]["train_times"]
        fp32_mem = data["fp32"]["peak_memory"]
        mixed_mem = data["mixed"]["peak_memory"]
    else:
        fp32_accs = data.get("fp32_test_accs", [])
        mixed_accs = data.get("mixed_test_accs", [])
        fp32_times = data.get("fp32_train_times", [])
        mixed_times = data.get("mixed_train_times", [])
        fp32_mem = data.get("fp32_peak_memory", [])
        mixed_mem = data.get("mixed_peak_memory", [])

    if not fp32_accs or not mixed_accs:
        print("No benchmark data found for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(
        ["FP32", "Mixed"],
        [np.mean(fp32_accs), np.mean(mixed_accs)],
        yerr=[np.std(fp32_accs), np.std(mixed_accs)],
        capsize=5,
        color=["#2196F3", "#4CAF50"],
    )
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Accuracy (higher is better)")

    if fp32_times and mixed_times:
        axes[1].bar(
            ["FP32", "Mixed"],
            [np.mean(fp32_times), np.mean(mixed_times)],
            color=["#2196F3", "#4CAF50"],
        )
        axes[1].set_ylabel("Training Time (seconds)")
        axes[1].set_title("Training Time (lower is better)")

    if fp32_mem and mixed_mem:
        axes[2].bar(
            ["FP32", "Mixed"],
            [np.mean(fp32_mem), np.mean(mixed_mem)],
            color=["#2196F3", "#4CAF50"],
        )
        axes[2].set_ylabel("Peak Memory (GB)")
        axes[2].set_title("Memory Usage (lower is better)")

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
