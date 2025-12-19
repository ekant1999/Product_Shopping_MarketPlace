# mixed_precision/utils/profiling.py
import jax
import time
import numpy as np


class GPUProfiler:
    def __init__(self):
        self.timings = {}

    def time_function(self, name, fn, *args, warmup=3, repeats=10):
        for _ in range(warmup):
            result = fn(*args)
            jax.block_until_ready(result)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn(*args)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t0)

        self.timings[name] = {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }
        return result

    def report(self):
        print(f"\n{'Operation':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12}")
        print("-" * 66)
        for name, t in sorted(
            self.timings.items(), key=lambda x: -x[1]["mean_ms"]
        ):
            print(
                f"{name:<30} {t['mean_ms']:<12.3f} {t['std_ms']:<12.3f} {t['min_ms']:<12.3f}"
            )
