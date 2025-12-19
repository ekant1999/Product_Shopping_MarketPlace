# Low-Precision Deep Learning on Next-Gen GPU Architectures

This project implements **mixed-precision deep learning** with custom CUDA kernels (CUTLASS) and JAX/Flax, targeting:

1. **~38% reduction in training memory** while keeping **within ~1.2% accuracy** on CIFAR-100.
2. **~31% reduction in end-to-end training time** for convolutional models over 50 runs.

## Tech Stack

- **C++17 / CUDA 11.8+** — custom kernels, CUTLASS GEMM and convolution
- **CUTLASS 2.x or 3.x** — mixed-precision GEMM with fused bias + ReLU
- **JAX 0.4+**, **Flax**, **Optax** — training, autograd, JIT
- **CIFAR-100** — 60k images, 100 classes, ResNet-18–style model
- **CMake + pybind11** — building and binding CUDA extensions to Python

## Quick Start

```bash
# 1. Virtual environment
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows

# 2. Python dependencies
pip install -r requirements.txt

# 3. Clone CUTLASS (required for custom GEMM/conv kernels)
mkdir -p third_party
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
cd third_party/cutlass && git checkout v3.3.0 && cd ../..

# 4. Build CUDA extension
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

# 5. Install package (so mp_kernels can be imported)
pip install -e .

# 6. Train FP32 baseline (single run)
python scripts/train_baseline.py --epochs 200 --lr 0.1 --batch-size 128 --seed 0

# 7. Train mixed-precision (single run)
python scripts/train_mixed_precision.py --epochs 200 --lr 0.1 --batch-size 128 --seed 0
```

## Project Layout

```
low-precision-dl/
├── CMakeLists.txt, setup.py, requirements.txt
├── csrc/                    # C++/CUDA sources
│   ├── kernels/             # Mixed-precision GEMM, fused linear+ReLU, fused conv2d
│   ├── loss_scaling/        # Dynamic loss scaler (GPU)
│   ├── utils/               # CUDA helpers, precision cast
│   └── bindings.cpp         # pybind11 → Python
├── mixed_precision/         # Python package
│   ├── data/                # CIFAR-100, augmentation
│   ├── models/              # ResNet, MixedPrecisionResNet
│   ├── training/            # FP32/mixed trainers, loss scaling, master weights
│   ├── kernels/             # JAX primitives, fused ops wrappers
│   └── utils/               # Metrics, profiling
├── scripts/                 # train_baseline, train_mixed_precision, benchmark, compare_results
├── configs/                 # default.yaml
├── tests/                   # Correctness and integration tests
└── results/                 # Benchmark outputs
```

## Benchmark (50 runs)

```bash
python scripts/benchmark.py --num-runs 50 --output results/benchmark_results.json
python scripts/compare_results.py --input results/benchmark_results.json --output results/comparison.png
```

## CUTLASS Note

The code is written for CUTLASS-style GEMM and epilogue APIs. If you use **CUTLASS 3.x**, include paths and type names may differ from CUTLASS 2.x; adjust `CMakeLists.txt` and kernel includes as needed. See [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) for your version.

## License

MIT-style or as specified in the project.
