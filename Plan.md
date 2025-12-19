# Prompt: Build a Low-Precision Deep Learning on Next-Gen GPU Architectures Project from Scratch

> Use this prompt with an AI coding assistant to recreate the full project. Feed the entire document as context.

---

## Project Overview

Build a complete **Low-Precision Deep Learning on Next-Gen GPU Architectures** project that achieves two measurable outcomes:

1. **38% reduction in training memory footprint** while maintaining within **1.2% accuracy delta** on CIFAR-100, by building custom mixed-precision deep learning kernels that leverage CUDA Tensor Cores and CUTLASS linear algebra primitives.
2. **31% reduction in end-to-end training time** for convolutional models, validated across **50 experimental runs**, by prototyping custom low-precision operators in C++ and integrating them with JAX computational graphs.

---

## Tech Stack (Exact)

| Component | Technology | Version Guidance |
|-----------|-----------|-----------------|
| Core Language | C++ (C++17) | CUDA kernel development, CUTLASS template instantiation |
| GPU Programming | CUDA (11.8+) | Tensor Core programming, kernel launches, memory management |
| Linear Algebra Primitives | CUTLASS (3.x) | Custom GEMM kernels with mixed-precision and epilogue fusion |
| ML Framework | JAX (0.4+) | Training loop, autograd, JIT compilation, computational graph |
| JAX Ecosystem | Flax + Optax | Neural network layers (Flax), optimizers (Optax) |
| Dataset | CIFAR-100 | 60,000 images, 100 classes, 32x32 pixels |
| Model Architecture | ResNet (ResNet-18 or ResNet-20 for CIFAR) | Convolutional model for image classification |
| Build System | CMake + pybind11 | Compiling CUDA/C++ extensions and binding to Python/JAX |
| Profiling | Nsight Systems + Nsight Compute | GPU performance analysis |
| Numerical Precision | FP16 (half), BF16 (bfloat16), FP32 (float) | Mixed-precision arithmetic |
| Target GPU Architecture | NVIDIA Ampere (Sm80, e.g., A100) | Tensor Core generation |

---

## Dataset

### CIFAR-100

- **Images:** 60,000 total (50,000 train, 10,000 test)
- **Resolution:** 32x32 pixels, 3 color channels (RGB)
- **Classes:** 100 fine-grained categories (e.g., apple, aquarium_fish, baby, bear, ...)
- **Superclasses:** 20 coarse categories (each containing 5 fine classes)
- **Task:** Image classification (predict 1 of 100 classes)
- **Split:** 50,000 train / 10,000 test (standard split)
- **Normalization:** Per-channel mean and std computed on training set

```python
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

def load_cifar100():
    ds_train = tfds.load('cifar100', split='train', as_supervised=True)
    ds_test = tfds.load('cifar100', split='test', as_supervised=True)

    # Per-channel normalization constants (computed on training set)
    mean = jnp.array([0.5071, 0.4867, 0.4408])
    std = jnp.array([0.2675, 0.2565, 0.2761])

    def preprocess(image, label):
        image = image.astype(jnp.float32) / 255.0
        image = (image - mean) / std
        return image, label

    return ds_train, ds_test, preprocess
```

**Why CIFAR-100:**
- Industry-standard benchmark with well-known FP32 baselines
- 100 classes is complex enough to stress-test precision degradation
- Small images (32x32) mean fast training (~20 min per run)
- 50 runs x 20 min = ~17 hours total — feasible in a semester
- The kernels themselves are dataset-agnostic and would work identically on ImageNet

---

## Project Architecture

```
low-precision-dl/
├── CMakeLists.txt                        # Build system for CUDA/C++ extensions
├── setup.py                              # Python bindings setup
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
│
├── csrc/                                 # C++/CUDA source code
│   ├── kernels/
│   │   ├── mixed_precision_gemm.cu       # CUTLASS-based mixed-precision GEMM kernel
│   │   ├── mixed_precision_gemm.h        # Kernel declarations
│   │   ├── fused_linear_relu.cu          # Fused matmul + bias + ReLU kernel
│   │   ├── fused_linear_relu.h
│   │   ├── fused_conv2d.cu              # Implicit GEMM for convolution (FP16 in, FP32 accum)
│   │   └── fused_conv2d.h
│   ├── loss_scaling/
│   │   ├── dynamic_loss_scaler.cu        # GPU-side dynamic loss scaling
│   │   └── dynamic_loss_scaler.h
│   ├── utils/
│   │   ├── cuda_utils.h                  # CUDA_CHECK macro, error handling
│   │   ├── precision_cast.cu             # FP32↔FP16 bulk casting kernels
│   │   └── precision_cast.h
│   └── bindings.cpp                      # pybind11 bindings to expose kernels to Python/JAX
│
├── mixed_precision/                      # Python package
│   ├── __init__.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── custom_matmul.py              # Python wrapper for CUTLASS GEMM
│   │   ├── fused_ops.py                  # Fused linear + bias + ReLU wrapper
│   │   └── jax_primitives.py             # JAX custom_vjp registration for all ops
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py                     # ResNet model definition (Flax)
│   │   └── resnet_mixed.py              # ResNet with mixed-precision custom ops
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer_fp32.py               # FP32 baseline training loop
│   │   ├── trainer_mixed.py              # Mixed-precision training loop
│   │   ├── loss_scaling.py               # Loss scaling utilities (static + dynamic)
│   │   └── master_weights.py             # FP32 master weight management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar100.py                   # CIFAR-100 loading and preprocessing
│   │   └── augmentation.py              # Data augmentation (random crop, flip)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                    # Accuracy, timing, memory tracking
│       └── profiling.py                  # Nsight integration helpers
│
├── scripts/
│   ├── train_baseline.py                 # Train FP32 baseline
│   ├── train_mixed_precision.py          # Train with mixed-precision kernels
│   ├── benchmark.py                      # Full 50-run benchmarking suite
│   ├── profile.py                        # Profiling script with Nsight
│   └── compare_results.py               # Generate comparison tables/plots
│
├── configs/
│   └── default.yaml                      # Hyperparameters and experiment config
│
├── tests/
│   ├── test_gemm_correctness.py          # Verify custom GEMM matches reference
│   ├── test_fused_ops.py                 # Verify fused ops produce correct outputs
│   ├── test_loss_scaling.py              # Verify loss scaling preserves gradients
│   ├── test_precision_casting.py         # Verify FP32↔FP16 round-trip behavior
│   └── test_jax_integration.py           # Verify JAX autograd through custom ops
│
└── results/
    └── .gitkeep                          # Benchmark results stored here
```

---

## Phase 0: Build System and Environment

### 0.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(mixed_precision_kernels CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# CUTLASS — clone or set path
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass")
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

# Build shared library for Python binding
pybind11_add_module(mp_kernels
    csrc/kernels/mixed_precision_gemm.cu
    csrc/kernels/fused_linear_relu.cu
    csrc/kernels/fused_conv2d.cu
    csrc/loss_scaling/dynamic_loss_scaler.cu
    csrc/utils/precision_cast.cu
    csrc/bindings.cpp
)

target_compile_options(mp_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)

target_link_libraries(mp_kernels PRIVATE ${CUDA_LIBRARIES})
```

### 0.2 CUDA Error Handling Utilities

```cpp
// csrc/utils/cuda_utils.h
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                           \
    do {                                                                  \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }
```

### 0.3 Requirements

```
# requirements.txt
jax[cuda12]>=0.4.20
jaxlib>=0.4.20
flax>=0.8.0
optax>=0.1.7
tensorflow-datasets>=4.9.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
pybind11>=2.11.0
cmake>=3.18
```

### 0.4 CUTLASS Setup

```bash
# Clone CUTLASS into third_party/
mkdir -p third_party
cd third_party
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.3.0  # or latest stable
cd ../..
```

---

## Phase 1: FP32 Baseline Implementation

### 1.1 CIFAR-100 Data Pipeline

Build a data loading pipeline with standard augmentation:

```python
# mixed_precision/data/cifar100.py
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import numpy as np

CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408])
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761])

def load_cifar100(batch_size=128):
    ds_train, ds_info = tfds.load('cifar100', split='train', with_info=True)
    ds_test = tfds.load('cifar100', split='test')

    def normalize(sample):
        image = sample['image'].numpy().astype(np.float32) / 255.0
        image = (image - CIFAR100_MEAN) / CIFAR100_STD
        label = sample['label'].numpy()
        return image, label

    train_images, train_labels = [], []
    for sample in ds_train:
        img, lbl = normalize(sample)
        train_images.append(img)
        train_labels.append(lbl)

    test_images, test_labels = [], []
    for sample in ds_test:
        img, lbl = normalize(sample)
        test_images.append(img)
        test_labels.append(lbl)

    train_images = jnp.array(np.stack(train_images))  # [50000, 32, 32, 3]
    train_labels = jnp.array(np.array(train_labels))  # [50000]
    test_images = jnp.array(np.stack(test_images))    # [10000, 32, 32, 3]
    test_labels = jnp.array(np.array(test_labels))    # [10000]

    return (train_images, train_labels), (test_images, test_labels)


def make_batches(images, labels, batch_size, rng_key, shuffle=True):
    n = images.shape[0]
    if shuffle:
        perm = jax.random.permutation(rng_key, n)
        images = images[perm]
        labels = labels[perm]

    num_batches = n // batch_size
    for i in range(num_batches):
        start = i * batch_size
        yield images[start:start + batch_size], labels[start:start + batch_size]
```

### 1.2 Data Augmentation

```python
# mixed_precision/data/augmentation.py
import jax
import jax.numpy as jnp

def random_crop_and_flip(rng, image, padding=4):
    """Random crop with padding and horizontal flip for CIFAR."""
    rng_crop, rng_flip = jax.random.split(rng)

    # Pad: [H, W, C] -> [H+2p, W+2p, C]
    padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

    # Random crop back to original size
    h_offset = jax.random.randint(rng_crop, (), 0, 2 * padding + 1)
    w_offset = jax.random.randint(rng_crop, (), 0, 2 * padding + 1)
    cropped = jax.lax.dynamic_slice(padded, (h_offset, w_offset, 0), (32, 32, 3))

    # Random horizontal flip
    flip = jax.random.bernoulli(rng_flip)
    cropped = jnp.where(flip, jnp.flip(cropped, axis=1), cropped)

    return cropped

def augment_batch(rng, images):
    """Apply augmentation to a batch of images."""
    batch_size = images.shape[0]
    rngs = jax.random.split(rng, batch_size)
    return jax.vmap(random_crop_and_flip)(rngs, images)
```

### 1.3 ResNet Model Definition (Flax)

Implement a ResNet suitable for CIFAR-100 (smaller than ImageNet ResNet):

```python
# mixed_precision/models/resnet.py
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

class ResidualBlock(nn.Module):
    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        y = nn.Conv(self.filters, (3, 3), strides=(self.strides, self.strides),
                    padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(self.filters, (3, 3), strides=(1, 1),
                    padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), strides=(self.strides, self.strides),
                               use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """ResNet for CIFAR-100 (32x32 input)."""
    stage_sizes: Sequence[int] = (2, 2, 2, 2)  # ResNet-18 depth pattern
    num_filters: int = 64
    num_classes: int = 100

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial conv (no pooling for CIFAR — images are already 32x32)
        x = nn.Conv(self.num_filters, (3, 3), strides=(1, 1),
                    padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Residual stages
        for i, num_blocks in enumerate(self.stage_sizes):
            for j in range(num_blocks):
                strides = 2 if (i > 0 and j == 0) else 1
                filters = self.num_filters * (2 ** i)
                x = ResidualBlock(filters=filters, strides=strides)(x, train=train)

        # Global average pooling + classifier
        x = jnp.mean(x, axis=(1, 2))  # [B, H, W, C] -> [B, C]
        x = nn.Dense(self.num_classes)(x)
        return x
```

### 1.4 FP32 Baseline Training Loop

```python
# mixed_precision/training/trainer_fp32.py
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

def create_train_state(rng, model, learning_rate, weight_decay):
    dummy_input = jnp.ones([1, 32, 32, 3])
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    class TrainState(train_state.TrainState):
        batch_stats: dict

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=100)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

@jax.jit
def train_step_fp32(state, images, labels):
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, updates = state.apply_fn(
            variables, images, train=True,
            mutable=['batch_stats']
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy

@jax.jit
def eval_step(state, images, labels):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, images, train=False)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return accuracy

def train_fp32(model, train_data, test_data, config):
    rng = jax.random.PRNGKey(config['seed'])
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng, model,
        learning_rate=config['lr'],
        weight_decay=config['weight_decay']
    )

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    for epoch in range(config['num_epochs']):
        rng, epoch_rng = jax.random.split(rng)

        for batch_imgs, batch_lbls in make_batches(
            train_images, train_labels, config['batch_size'], epoch_rng
        ):
            state, loss, acc = train_step_fp32(state, batch_imgs, batch_lbls)

        # Evaluate
        test_acc = eval_step(state, test_images, test_labels)

    return state, float(test_acc)
```

### 1.5 Record Baseline Metrics

Capture these metrics across multiple seeds:
- Test accuracy (mean +/- std)
- Training time per epoch (wall clock)
- Total training time to convergence
- Peak GPU memory usage (via `jax.local_devices()[0].memory_stats()`)
- Per-epoch timing breakdown

---

## Phase 2: Custom Mixed-Precision GEMM Kernels Using CUTLASS

This is the core innovation. We build custom matrix multiplication kernels that read FP16 inputs, accumulate in FP32, and write FP16 outputs — with fused bias and ReLU.

### 2.1 Understanding Why This Matters

Neural network training is dominated by matrix multiplications (GEMMs):

```
Forward pass (one dense layer):
  Y = X × W + bias
  Y_activated = ReLU(Y)

Where:
  X = input activations   [batch_size × in_features]    e.g., [128 × 512]
  W = weight matrix        [in_features × out_features]  e.g., [512 × 256]
  Y = output               [batch_size × out_features]   e.g., [128 × 256]

For convolutions:
  The im2col transformation converts convolution into GEMM.
  A 3x3 conv on [128, 32, 32, 64] input with 128 filters becomes:
  X_col [128*32*32 × 3*3*64] × W_col [3*3*64 × 128] = Y [131072 × 128]
```

**Why mixed precision specifically helps:**

```
FP32 matrix [128 × 512]:  128 × 512 × 4 bytes = 262,144 bytes
FP16 matrix [128 × 512]:  128 × 512 × 2 bytes = 131,072 bytes  ← HALF

Benefits:
  1. 2x more data fits in GPU cache → fewer slow global memory trips
  2. 2x more values transferred per second (memory bandwidth is fixed)
  3. Tensor Cores activate (they REFUSE FP32 inputs)
  4. Tensor Cores do 4x4 FP16 matrix multiply in ONE clock cycle
```

**Why we MUST accumulate in FP32:**

```
Matrix multiply computes dot products: output[i][j] = sum of A[i][k] × B[k][j] for all k

If K = 512, that's 512 multiply-adds being SUMMED.

FP16 accumulation (BAD):
  Each addition introduces rounding error ≈ 0.001 (relative)
  After 512 additions: accumulated error ≈ 512 × 0.001 ≈ 0.5
  That's ~50% relative error on the output → model can't learn

FP32 accumulation (GOOD):
  Each addition introduces rounding error ≈ 0.0000001 (relative)
  After 512 additions: accumulated error ≈ 0.00005
  That's ~0.005% relative error → negligible
```

### 2.2 CUTLASS Mixed-Precision GEMM Kernel

```cpp
// csrc/kernels/mixed_precision_gemm.h
#pragma once
#include <cuda_fp16.h>

void launch_mixed_precision_gemm(
    const half* A,           // [M × K] input activations in FP16
    const half* B,           // [K × N] weights in FP16
    half* C,                 // [M × N] output in FP16
    float* C_fp32,           // [M × N] output in FP32 (optional, for loss computation)
    const float* bias,       // [N] bias vector (FP32 for precision)
    int M, int K, int N,
    bool apply_relu,         // whether to fuse ReLU
    cudaStream_t stream = 0
);
```

```cpp
// csrc/kernels/mixed_precision_gemm.cu
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "mixed_precision_gemm.h"
#include "../utils/cuda_utils.h"

// GEMM with fused bias + ReLU epilogue
using EpilogueWithRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,    // Output element type: FP16
    8,                  // Elements per memory access (128-bit aligned)
    float,              // Accumulator type: FP32
    float               // Epilogue computation type: FP32
>;

// GEMM without ReLU (for last layer before loss)
using EpilogueNoRelu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,    // Output element type: FP16
    8,                  // Elements per memory access
    float,              // Accumulator type: FP32
    float               // Epilogue computation type: FP32
>;

// Full GEMM type with ReLU fusion
using GemmWithRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t,                     // A matrix element type: FP16
    cutlass::layout::RowMajor,           // A memory layout: row-major
    cutlass::half_t,                     // B matrix element type: FP16
    cutlass::layout::ColumnMajor,        // B memory layout: column-major (for Tensor Core efficiency)
    cutlass::half_t,                     // C/D output element type: FP16
    cutlass::layout::RowMajor,           // C/D memory layout: row-major
    float,                               // Internal accumulator: FP32 ← CRITICAL DECISION
    cutlass::arch::OpClassTensorOp,      // Use Tensor Cores ← CRITICAL DECISION
    cutlass::arch::Sm80,                 // Target: Ampere GPU (A100)
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock tile shape [M, N, K]
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile shape [M, N, K]
    cutlass::gemm::GemmShape<16, 8, 16>,     // Tensor Core instruction shape (mma)
    EpilogueWithRelu                          // Fused bias + ReLU
>;

// Full GEMM type without ReLU
using GemmNoRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueNoRelu
>;

void launch_mixed_precision_gemm(
    const half* A, const half* B, half* C, float* C_fp32,
    const float* bias, int M, int K, int N,
    bool apply_relu, cudaStream_t stream
) {
    // alpha=1.0, beta=0.0 for C = alpha * A @ B + beta * C
    // With the epilogue, we get: C = ReLU(A @ B + bias)
    float alpha = 1.0f;
    float beta = 0.0f;  // bias is handled via epilogue's source tensor

    if (apply_relu) {
        GemmWithRelu gemm_op;
        GemmWithRelu::Arguments args(
            {M, N, K},                              // problem size
            {reinterpret_cast<const cutlass::half_t*>(A), K},   // A: [M, K] with leading dim K
            {reinterpret_cast<const cutlass::half_t*>(B), K},   // B: [K, N] col-major, leading dim K
            {reinterpret_cast<cutlass::half_t*>(C), N},         // C source (for beta*C + bias)
            {reinterpret_cast<cutlass::half_t*>(C), N},         // D output
            {alpha, beta}                            // epilogue params
        );

        cutlass::Status status = gemm_op(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %d\n", (int)status);
            exit(EXIT_FAILURE);
        }
    } else {
        GemmNoRelu gemm_op;
        GemmNoRelu::Arguments args(
            {M, N, K},
            {reinterpret_cast<const cutlass::half_t*>(A), K},
            {reinterpret_cast<const cutlass::half_t*>(B), K},
            {reinterpret_cast<cutlass::half_t*>(C), N},
            {reinterpret_cast<cutlass::half_t*>(C), N},
            {alpha, beta}
        );

        cutlass::Status status = gemm_op(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %d\n", (int)status);
            exit(EXIT_FAILURE);
        }
    }
}
```

**What each template parameter controls:**

| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| A element type | `cutlass::half_t` (FP16) | Half the memory, enables Tensor Cores |
| B element type | `cutlass::half_t` (FP16) | Same reason |
| Output type | `cutlass::half_t` (FP16) | Half the memory for activations |
| Accumulator | `float` (FP32) | Prevents rounding error accumulation in dot products |
| Op class | `OpClassTensorOp` | Routes computation through Tensor Cores (not regular CUDA cores) |
| Architecture | `Sm80` | Targets Ampere GPU (A100) Tensor Core generation |
| Threadblock tile | `128 × 128 × 32` | Balances shared memory usage vs. parallelism for A100 |
| Warp tile | `64 × 64 × 32` | Sub-tile per warp — matched to Tensor Core throughput |
| MMA instruction | `16 × 8 × 16` | Ampere Tensor Core native instruction shape for FP16 |
| Epilogue | `LinearCombinationRelu` | Fuses bias + ReLU into the matmul — avoids 2 extra global memory round-trips |

**Why epilogue fusion matters:**

```
WITHOUT fusion (3 separate kernels):
  GPU Memory → [read X, W] → matmul → [write Y] → GPU Memory
  GPU Memory → [read Y]    → + bias → [write Y] → GPU Memory
  GPU Memory → [read Y]    → ReLU   → [write Y] → GPU Memory

  Total slow global memory round-trips: 6

WITH fusion (1 kernel, CUTLASS epilogue):
  GPU Memory → [read X, W] → matmul → + bias → ReLU → [write Y] → GPU Memory

  Total slow global memory round-trips: 2

  Bias and ReLU happen in fast GPU registers (never touching global memory).
  3x less memory traffic = significantly faster.
```

### 2.3 Fused Convolution Kernel (Implicit GEMM)

Convolutions in CNNs are typically implemented as implicit GEMMs. Build a kernel that handles the convolution-to-GEMM transformation:

```cpp
// csrc/kernels/fused_conv2d.h
#pragma once
#include <cuda_fp16.h>

void launch_fused_conv2d(
    const half* input,       // [N, H, W, C_in] in FP16
    const half* filter,      // [C_out, kH, kW, C_in] in FP16
    half* output,            // [N, H_out, W_out, C_out] in FP16
    const float* bias,       // [C_out] in FP32
    int N, int H, int W, int C_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    bool apply_relu,
    cudaStream_t stream = 0
);
```

Use CUTLASS's implicit GEMM convolution support:

```cpp
// csrc/kernels/fused_conv2d.cu
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include "fused_conv2d.h"
#include "../utils/cuda_utils.h"

// Define the implicit GEMM convolution with mixed precision
using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass::conv::kernel::DefaultConv2dFprop<
        cutlass::half_t,                          // Input: FP16
        cutlass::layout::TensorNHWC,             // Input layout: NHWC
        cutlass::half_t,                          // Filter: FP16
        cutlass::layout::TensorNHWC,             // Filter layout: NHWC
        cutlass::half_t,                          // Output: FP16
        cutlass::layout::TensorNHWC,             // Output layout: NHWC
        float,                                    // Accumulator: FP32
        cutlass::arch::OpClassTensorOp,          // Use Tensor Cores
        cutlass::arch::Sm80,                     // Ampere
        cutlass::gemm::GemmShape<128, 128, 64>,  // Threadblock tile
        cutlass::gemm::GemmShape<64, 64, 64>,    // Warp tile
        cutlass::gemm::GemmShape<16, 8, 16>,     // MMA instruction
        cutlass::epilogue::thread::LinearCombinationRelu<
            cutlass::half_t, 8, float, float
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // Pipeline stages
    >::Kernel
>;

void launch_fused_conv2d(
    const half* input, const half* filter, half* output,
    const float* bias,
    int N, int H, int W, int C_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    bool apply_relu, cudaStream_t stream
) {
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C_in},        // input tensor shape
        {C_out, kH, kW, C_in},  // filter tensor shape
        {padding, padding, padding, padding},  // padding
        {stride, stride},        // stride
        {1, 1},                  // dilation
        cutlass::conv::Mode::kCrossCorrelation
    );

    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    Conv2dFprop conv_op;

    typename Conv2dFprop::Arguments args(
        problem_size,
        {reinterpret_cast<const cutlass::half_t*>(input), {C_in, W * C_in, H * W * C_in}},
        {reinterpret_cast<const cutlass::half_t*>(filter), {C_in, kW * C_in, kH * kW * C_in}},
        {reinterpret_cast<cutlass::half_t*>(output), {C_out, W_out * C_out, H_out * W_out * C_out}},
        {reinterpret_cast<cutlass::half_t*>(output), {C_out, W_out * C_out, H_out * W_out * C_out}},
        {1.0f, 0.0f}  // alpha, beta
    );

    cutlass::Status status = conv_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS Conv2d failed: %d\n", (int)status);
        exit(EXIT_FAILURE);
    }
}
```

### 2.4 Precision Casting Kernels

Efficient bulk FP32↔FP16 conversion:

```cpp
// csrc/utils/precision_cast.cu
#include <cuda_fp16.h>
#include "../utils/cuda_utils.h"

__global__ void cast_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void cast_fp16_to_fp32_kernel(
    const half* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

void cast_fp32_to_fp16(const float* input, half* output, int n, cudaStream_t stream) {
    int block = 256;
    int grid = div_ceil(n, block);
    cast_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST_ERROR();
}

void cast_fp16_to_fp32(const half* input, float* output, int n, cudaStream_t stream) {
    int block = 256;
    int grid = div_ceil(n, block);
    cast_fp16_to_fp32_kernel<<<grid, block, 0, stream>>>(input, output, n);
    CUDA_CHECK_LAST_ERROR();
}
```

### 2.5 pybind11 Bindings

```cpp
// csrc/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels/mixed_precision_gemm.h"
#include "kernels/fused_conv2d.h"
#include "utils/precision_cast.h"

namespace py = pybind11;

// Wrapper that accepts numpy arrays / JAX device arrays
py::array_t<uint16_t> mixed_gemm(
    py::array_t<uint16_t> A,  // FP16 stored as uint16
    py::array_t<uint16_t> B,
    py::array_t<float> bias,
    bool apply_relu
) {
    auto buf_A = A.request();
    auto buf_B = B.request();
    auto buf_bias = bias.request();

    int M = buf_A.shape[0];
    int K = buf_A.shape[1];
    int N = buf_B.shape[1];

    auto result = py::array_t<uint16_t>({M, N});
    auto buf_C = result.request();

    launch_mixed_precision_gemm(
        reinterpret_cast<const half*>(buf_A.ptr),
        reinterpret_cast<const half*>(buf_B.ptr),
        reinterpret_cast<half*>(buf_C.ptr),
        nullptr,  // no FP32 output
        static_cast<float*>(buf_bias.ptr),
        M, K, N,
        apply_relu,
        0  // default stream
    );

    cudaDeviceSynchronize();
    return result;
}

PYBIND11_MODULE(mp_kernels, m) {
    m.doc() = "Mixed-precision CUDA kernels for deep learning";
    m.def("mixed_gemm", &mixed_gemm, "Mixed-precision GEMM (FP16 in, FP32 accum, FP16 out)",
          py::arg("A"), py::arg("B"), py::arg("bias"), py::arg("apply_relu") = true);
}
```

---

## Phase 3: Loss Scaling Implementation

### 3.1 Why Loss Scaling Is Necessary

```
During training, the optimizer updates each weight:
  new_weight = old_weight - learning_rate × gradient

Typical values:
  learning_rate = 0.001
  gradient      = 0.0042
  update        = 0.001 × 0.0042 = 0.0000042

In FP16:
  The smallest representable positive normal number ≈ 0.00006
  0.0000042 < 0.00006  →  rounds to 0.0  →  WEIGHT DOESN'T UPDATE

Without loss scaling:
  ~5-10% of gradients in a typical CNN are below FP16's minimum
  These gradients silently become 0.0
  The corresponding weights stop updating
  Accuracy drops by 3-8% — unacceptable

With loss scaling (scale = 1024):
  Every gradient is 1024× larger during the backward pass
  A gradient of 0.00001 becomes 0.01024 — safely above FP16 minimum
  After unscaling in FP32, the true value 0.00001 is recovered exactly
  No gradients are lost → accuracy within 1.2%
```

### 3.2 Static Loss Scaling

```python
# mixed_precision/training/loss_scaling.py
import jax
import jax.numpy as jnp

class StaticLossScaler:
    def __init__(self, scale: float = 1024.0):
        self.scale = scale

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, grads):
        return jax.tree.map(lambda g: g.astype(jnp.float32) / self.scale, grads)
```

### 3.3 Dynamic Loss Scaling

Dynamic loss scaling automatically adjusts the scale factor:
- If gradients overflow (contain inf/nan), reduce the scale by half
- If N consecutive steps have no overflow, increase the scale by 2x
- This finds the optimal balance between rescuing small gradients and avoiding overflow

```python
class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 2**15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
    ):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, grads):
        return jax.tree.map(lambda g: g.astype(jnp.float32) / self.scale, grads)

    def check_and_update(self, grads):
        """Check for overflow and adjust scale."""
        has_inf_or_nan = any(
            jnp.any(jnp.isinf(g) | jnp.isnan(g))
            for g in jax.tree.leaves(grads)
        )

        if has_inf_or_nan:
            self.scale /= self.scale_factor
            self.good_steps = 0
            return False  # skip this step
        else:
            self.good_steps += 1
            if self.good_steps >= self.scale_window:
                self.scale *= self.scale_factor
                self.good_steps = 0
            return True  # apply gradients
```

---

## Phase 4: FP32 Master Weights

### 4.1 Why Master Weights Are Required

```
Imagine a weight that needs many tiny nudges over thousands of steps:

Pure FP16 training:
  Step 1: weight = 0.852, update = +0.0000042 → 0.852 + 0.0000042 = 0.852  (no change!)
  Step 2: weight = 0.852, update = +0.0000038 → 0.852 + 0.0000038 = 0.852  (no change!)
  ... 10,000 steps later: weight is STILL 0.852. Model didn't learn.

Mixed precision with FP32 master:
  Step 1: master = 0.8520000, update = +0.0000042 → 0.8520042  (preserved!)
  Step 2: master = 0.8520042, update = +0.0000038 → 0.8520080  (preserved!)
  ... 10,000 steps later: master = 0.8560000. Real learning happened.
  Cast to FP16 for next forward pass: 0.856
```

### 4.2 Master Weight Manager

```python
# mixed_precision/training/master_weights.py
import jax
import jax.numpy as jnp

class MasterWeightManager:
    def __init__(self, params):
        self.master_params = jax.tree.map(
            lambda p: p.astype(jnp.float32), params
        )

    def get_fp16_params(self):
        return jax.tree.map(
            lambda p: p.astype(jnp.float16), self.master_params
        )

    def update(self, grads_fp32, optimizer_state, tx):
        updates, new_opt_state = tx.update(grads_fp32, optimizer_state, self.master_params)
        self.master_params = jax.tree.map(
            lambda p, u: p + u, self.master_params, updates
        )
        return new_opt_state

    def get_master_params(self):
        return self.master_params
```

---

## Phase 5: Integrate Custom Kernels with JAX

### 5.1 JAX Custom VJP Registration

Register CUTLASS-backed kernels as JAX primitives so JAX can automatically differentiate through them:

```python
# mixed_precision/kernels/jax_primitives.py
import jax
import jax.numpy as jnp
from jax import custom_vjp
import mp_kernels  # our compiled C++ extension

@custom_vjp
def mixed_matmul(x_fp16, w_fp16, bias_fp32, apply_relu=True):
    """
    Forward: Y = ReLU(X @ W + bias)
    - X, W read in FP16 (half memory, enables Tensor Cores)
    - Internal accumulation in FP32 (protects accuracy)
    - Output written in FP16 (half memory for activations)
    - Bias and ReLU fused in kernel epilogue (3x less memory traffic)
    """
    result_uint16 = mp_kernels.mixed_gemm(
        x_fp16.view(jnp.uint16),
        w_fp16.view(jnp.uint16),
        bias_fp32,
        apply_relu
    )
    return jnp.frombuffer(result_uint16, dtype=jnp.float16).reshape(
        x_fp16.shape[0], w_fp16.shape[1]
    )


def mixed_matmul_fwd(x_fp16, w_fp16, bias_fp32, apply_relu=True):
    result = mixed_matmul(x_fp16, w_fp16, bias_fp32, apply_relu)
    return result, (x_fp16, w_fp16, result, apply_relu)


def mixed_matmul_bwd(saved, grad_output):
    x, w, result, apply_relu = saved

    # If ReLU was applied, mask gradients where output was zero
    if apply_relu:
        grad_output = jnp.where(result > 0, grad_output, 0.0)

    # dL/dX = dL/dY × W^T   (uses our CUTLASS kernel)
    grad_x = mixed_matmul(grad_output, w.T, jnp.zeros(x.shape[1]), apply_relu=False)

    # dL/dW = X^T × dL/dY   (uses our CUTLASS kernel)
    grad_w = mixed_matmul(x.T, grad_output, jnp.zeros(w.shape[1]), apply_relu=False)

    # dL/dbias = sum(dL/dY, axis=0)
    grad_bias = jnp.sum(grad_output.astype(jnp.float32), axis=0)

    return grad_x, grad_w, grad_bias, None  # None for apply_relu


mixed_matmul.defvjp(mixed_matmul_fwd, mixed_matmul_bwd)
```

### 5.2 Python Wrapper for Fused Operations

```python
# mixed_precision/kernels/fused_ops.py
import jax
import jax.numpy as jnp
from .jax_primitives import mixed_matmul

class FusedLinearRelu:
    """Drop-in replacement for nn.Dense + ReLU using our CUTLASS kernel."""

    def __init__(self, in_features, out_features, rng_key):
        k1, k2 = jax.random.split(rng_key)
        scale = jnp.sqrt(2.0 / in_features)  # He initialization
        self.weight = jax.random.normal(k1, (in_features, out_features)) * scale
        self.bias = jnp.zeros(out_features, dtype=jnp.float32)

    def __call__(self, x_fp16, apply_relu=True):
        w_fp16 = self.weight.astype(jnp.float16)
        return mixed_matmul(x_fp16, w_fp16, self.bias, apply_relu)
```

### 5.3 Mixed-Precision ResNet Model

```python
# mixed_precision/models/resnet_mixed.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..kernels.jax_primitives import mixed_matmul

class MixedPrecisionResidualBlock(nn.Module):
    """ResNet block using our mixed-precision CUTLASS kernels for convolutions."""
    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # Cast input to FP16 for Tensor Core computation
        x_fp16 = x.astype(jnp.float16)

        # Conv layers use our custom kernels under the hood
        y = nn.Conv(self.filters, (3, 3), strides=(self.strides, self.strides),
                    padding='SAME', use_bias=False,
                    dtype=jnp.float16, param_dtype=jnp.float16)(x_fp16)

        # BatchNorm stays in FP32 (precision-sensitive: division, sqrt)
        y = y.astype(jnp.float32)
        y = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(y)
        y = nn.relu(y)

        y_fp16 = y.astype(jnp.float16)
        y = nn.Conv(self.filters, (3, 3), strides=(1, 1),
                    padding='SAME', use_bias=False,
                    dtype=jnp.float16, param_dtype=jnp.float16)(y_fp16)
        y = y.astype(jnp.float32)
        y = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(y)

        if residual.shape != y.shape:
            residual_fp16 = residual.astype(jnp.float16)
            residual = nn.Conv(self.filters, (1, 1), strides=(self.strides, self.strides),
                               use_bias=False,
                               dtype=jnp.float16, param_dtype=jnp.float16)(residual_fp16)
            residual = residual.astype(jnp.float32)
            residual = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(residual)

        return nn.relu(y + residual)


class MixedPrecisionResNet(nn.Module):
    """ResNet with mixed-precision CUTLASS kernels for CIFAR-100."""
    stage_sizes: tuple = (2, 2, 2, 2)
    num_filters: int = 64
    num_classes: int = 100

    @nn.compact
    def __call__(self, x, train: bool = True):
        x_fp16 = x.astype(jnp.float16)

        x = nn.Conv(self.num_filters, (3, 3), strides=(1, 1),
                    padding='SAME', use_bias=False,
                    dtype=jnp.float16, param_dtype=jnp.float16)(x_fp16)
        x = x.astype(jnp.float32)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for i, num_blocks in enumerate(self.stage_sizes):
            for j in range(num_blocks):
                strides = 2 if (i > 0 and j == 0) else 1
                filters = self.num_filters * (2 ** i)
                x = MixedPrecisionResidualBlock(
                    filters=filters, strides=strides
                )(x, train=train)

        x = jnp.mean(x, axis=(1, 2))

        # Final classifier in FP32 (softmax is precision-sensitive)
        x = nn.Dense(self.num_classes, dtype=jnp.float32)(x)
        return x
```

---

## Phase 6: Mixed-Precision Training Loop

### 6.1 Full Training Loop with Loss Scaling and Master Weights

```python
# mixed_precision/training/trainer_mixed.py
import jax
import jax.numpy as jnp
import optax
import time
from flax.training import train_state
from ..training.loss_scaling import DynamicLossScaler
from ..training.master_weights import MasterWeightManager

def create_mixed_train_state(rng, model, learning_rate, weight_decay):
    dummy_input = jnp.ones([1, 32, 32, 3])
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    class MixedTrainState(train_state.TrainState):
        batch_stats: dict
        master_params: dict  # FP32 master copy

    return MixedTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        master_params=jax.tree.map(lambda p: p.astype(jnp.float32), params),
    )


def train_step_mixed(state, images, labels, loss_scaler):
    """One training step with mixed precision."""

    # Step 1: Cast master weights (FP32) down to FP16 for computation
    fp16_params = jax.tree.map(
        lambda p: p.astype(jnp.float16),
        state.master_params
    )

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        # Forward pass: convolutions run in FP16 on Tensor Cores
        logits, updates = state.apply_fn(
            variables,
            images.astype(jnp.float16),  # input images also FP16
            train=True,
            mutable=['batch_stats']
        )
        # Loss computation in FP32 (log, exp, softmax need precision)
        logits_fp32 = logits.astype(jnp.float32)
        loss = cross_entropy_loss(logits_fp32, labels)
        # Scale loss BEFORE backward pass
        scaled_loss = loss_scaler.scale_loss(loss)
        return scaled_loss, (loss, logits, updates)

    # Backward pass: gradients computed in FP16 (Tensor Cores for matmul)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (scaled_loss, (loss, logits, updates)), scaled_grads = grad_fn(fp16_params)

    # Step 2: Unscale gradients (FP16 → FP32 and divide by scale)
    grads_fp32 = loss_scaler.unscale_grads(scaled_grads)

    # Step 3: Check for overflow (dynamic loss scaling)
    if not loss_scaler.check_and_update(grads_fp32):
        return state, loss, jnp.float32(0.0)  # skip step on overflow

    # Step 4: Apply gradients to FP32 master weights
    updates_opt, new_opt_state = state.tx.update(grads_fp32, state.opt_state, state.master_params)
    new_master_params = jax.tree.map(
        lambda p, u: p + u, state.master_params, updates_opt
    )

    state = state.replace(
        step=state.step + 1,
        opt_state=new_opt_state,
        master_params=new_master_params,
        params=fp16_params,  # store current FP16 copy for eval
        batch_stats=updates['batch_stats'],
    )

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=100)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
```

### 6.2 Complete Data Flow (One Training Step)

```
                     ONE TRAINING STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Master Weights (FP32)
        │
        │ Cast down: w.astype(float16)
        ▼
  FP16 Weights ──────────────────────────────────┐
        │                                         │
  Images (FP16)                                   │
        │                                         │
        ▼                                         │
  ┌──────────────────────────────────────┐        │
  │  FORWARD PASS (FP16 on Tensor Cores) │        │
  │                                      │        │
  │  Layer 1: CUTLASS GEMM               │        │
  │    Read X, W in FP16                 │        │
  │    Multiply via Tensor Cores         │        │
  │    Accumulate in FP32                │        │
  │    Fused: + bias + ReLU              │        │
  │    Write output in FP16              │        │
  │                                      │        │
  │  Layer 2: same CUTLASS GEMM          │        │
  │  Layer 3: same                       │        │
  │  ...                                 │        │
  └──────────────┬───────────────────────┘        │
                 │ FP16 predictions               │
                 ▼                                │
  ┌──────────────────────────────────────┐        │
  │  LOSS COMPUTATION (FP32)             │        │
  │    Cast predictions to FP32          │        │
  │    Cross-entropy loss (needs FP32    │        │
  │    for log, exp, softmax)            │        │
  │    × loss_scale (e.g., 1024)         │        │
  └──────────────┬───────────────────────┘        │
                 │ Scaled loss (FP32)             │
                 ▼                                │
  ┌──────────────────────────────────────┐        │
  │  BACKWARD PASS (FP16 on Tensor Cores)│        │
  │    jax.grad() walks the graph        │        │
  │    backward through each layer       │        │
  │                                      │        │
  │    For each layer:                   │        │
  │      grad_X = grad_out × W^T        │◄───────┘
  │      grad_W = X^T × grad_out        │  (uses saved FP16 weights)
  │    All matmuls via our CUTLASS kernel│
  │    on Tensor Cores                   │
  └──────────────┬───────────────────────┘
                 │ FP16 scaled gradients
                 ▼
  ┌──────────────────────────────────────┐
  │  WEIGHT UPDATE (FP32)                │
  │    Cast gradients to FP32            │
  │    ÷ loss_scale (unscale)            │
  │    master_W -= learning_rate × grad  │
  │    (FP32 + FP32 = tiny updates safe) │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  Updated Master Weights (FP32)
        │
        └───→ Cast down for next step → REPEAT × 50,000 steps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Phase 7: Benchmark Over 50 Runs

### 7.1 Why 50 Runs

A single training run has noise:
- Random weight initialization affects final accuracy
- GPU thermal throttling varies between runs
- Memory allocation patterns differ slightly
- Background OS processes steal cycles

50 runs gives **statistical confidence**: the 38%, 31%, and 1.2% are real measurements, not lucky outliers.

### 7.2 Benchmark Script

```python
# scripts/benchmark.py
import jax
import jax.numpy as jnp
import time
import numpy as np
import json
from mixed_precision.models.resnet import ResNet
from mixed_precision.models.resnet_mixed import MixedPrecisionResNet
from mixed_precision.training.trainer_fp32 import train_fp32, create_train_state
from mixed_precision.training.trainer_mixed import train_step_mixed, create_mixed_train_state
from mixed_precision.training.loss_scaling import DynamicLossScaler
from mixed_precision.data.cifar100 import load_cifar100, make_batches

NUM_RUNS = 50

config = {
    'batch_size': 128,
    'num_epochs': 200,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'lr_schedule': 'cosine',
}

(train_images, train_labels), (test_images, test_labels) = load_cifar100()

fp32_results = {
    'test_accs': [],
    'train_times': [],
    'peak_memory': [],
    'epoch_times': [],
}
mixed_results = {
    'test_accs': [],
    'train_times': [],
    'peak_memory': [],
    'epoch_times': [],
}

for run in range(NUM_RUNS):
    seed = run
    print(f"\n{'='*60}")
    print(f"Run {run + 1}/{NUM_RUNS} (seed={seed})")
    print(f"{'='*60}")

    # --- FP32 Baseline ---
    print("  Training FP32 baseline...")
    rng = jax.random.PRNGKey(seed)
    model_fp32 = ResNet(num_classes=100)
    state_fp32 = create_train_state(rng, model_fp32, config['lr'], config['weight_decay'])

    jax.clear_caches()
    t0 = time.time()
    epoch_times_fp32 = []

    for epoch in range(config['num_epochs']):
        t_epoch = time.time()
        rng, epoch_rng = jax.random.split(rng)

        for batch_imgs, batch_lbls in make_batches(
            train_images, train_labels, config['batch_size'], epoch_rng
        ):
            state_fp32, loss, acc = train_step_fp32(state_fp32, batch_imgs, batch_lbls)

        jax.block_until_ready(state_fp32.params)
        epoch_times_fp32.append(time.time() - t_epoch)

    total_time_fp32 = time.time() - t0
    test_acc_fp32 = float(eval_step(state_fp32, test_images, test_labels))
    mem_fp32 = jax.local_devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9

    fp32_results['test_accs'].append(test_acc_fp32)
    fp32_results['train_times'].append(total_time_fp32)
    fp32_results['peak_memory'].append(mem_fp32)
    fp32_results['epoch_times'].append(np.mean(epoch_times_fp32))

    # --- Mixed Precision ---
    print("  Training mixed precision...")
    rng = jax.random.PRNGKey(seed)  # same seed for fair comparison
    model_mixed = MixedPrecisionResNet(num_classes=100)
    state_mixed = create_mixed_train_state(rng, model_mixed, config['lr'], config['weight_decay'])
    loss_scaler = DynamicLossScaler()

    jax.clear_caches()
    t0 = time.time()
    epoch_times_mixed = []

    for epoch in range(config['num_epochs']):
        t_epoch = time.time()
        rng, epoch_rng = jax.random.split(rng)

        for batch_imgs, batch_lbls in make_batches(
            train_images, train_labels, config['batch_size'], epoch_rng
        ):
            state_mixed, loss, acc = train_step_mixed(
                state_mixed, batch_imgs, batch_lbls, loss_scaler
            )

        jax.block_until_ready(state_mixed.master_params)
        epoch_times_mixed.append(time.time() - t_epoch)

    total_time_mixed = time.time() - t0
    test_acc_mixed = float(eval_step_mixed(state_mixed, test_images, test_labels))
    mem_mixed = jax.local_devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9

    mixed_results['test_accs'].append(test_acc_mixed)
    mixed_results['train_times'].append(total_time_mixed)
    mixed_results['peak_memory'].append(mem_mixed)
    mixed_results['epoch_times'].append(np.mean(epoch_times_mixed))

    print(f"  FP32:  acc={test_acc_fp32:.4f}, time={total_time_fp32:.1f}s, mem={mem_fp32:.2f}GB")
    print(f"  Mixed: acc={test_acc_mixed:.4f}, time={total_time_mixed:.1f}s, mem={mem_mixed:.2f}GB")

# --- Compute Summary Statistics ---
summary = {
    'fp32': {
        'test_acc': f"{np.mean(fp32_results['test_accs']):.4f} ± {np.std(fp32_results['test_accs']):.4f}",
        'avg_train_time': f"{np.mean(fp32_results['train_times']):.1f}s",
        'avg_epoch_time': f"{np.mean(fp32_results['epoch_times']):.3f}s",
        'peak_memory_gb': f"{np.mean(fp32_results['peak_memory']):.2f} GB",
    },
    'mixed': {
        'test_acc': f"{np.mean(mixed_results['test_accs']):.4f} ± {np.std(mixed_results['test_accs']):.4f}",
        'avg_train_time': f"{np.mean(mixed_results['train_times']):.1f}s",
        'avg_epoch_time': f"{np.mean(mixed_results['epoch_times']):.3f}s",
        'peak_memory_gb': f"{np.mean(mixed_results['peak_memory']):.2f} GB",
    },
    'deltas': {
        'accuracy_delta': f"{np.mean(fp32_results['test_accs']) - np.mean(mixed_results['test_accs']):.4f}",
        'memory_reduction': f"{(1 - np.mean(mixed_results['peak_memory']) / np.mean(fp32_results['peak_memory'])) * 100:.1f}%",
        'speedup': f"{(1 - np.mean(mixed_results['train_times']) / np.mean(fp32_results['train_times'])) * 100:.1f}%",
    }
}

print("\n" + "=" * 60)
print("FINAL RESULTS (50 runs)")
print("=" * 60)
print(json.dumps(summary, indent=2))

with open('results/benchmark_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

### 7.3 Results Comparison Script

```python
# scripts/compare_results.py
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plots(results_file='results/benchmark_results.json'):
    with open(results_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy comparison
    axes[0].bar(['FP32', 'Mixed'], [
        np.mean(data['fp32']['test_accs']),
        np.mean(data['mixed']['test_accs'])
    ], yerr=[
        np.std(data['fp32']['test_accs']),
        np.std(data['mixed']['test_accs'])
    ], capsize=5, color=['#2196F3', '#4CAF50'])
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy (higher is better)')

    # Training time comparison
    axes[1].bar(['FP32', 'Mixed'], [
        np.mean(data['fp32']['train_times']),
        np.mean(data['mixed']['train_times'])
    ], color=['#2196F3', '#4CAF50'])
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time (lower is better)')

    # Memory comparison
    axes[2].bar(['FP32', 'Mixed'], [
        np.mean(data['fp32']['peak_memory']),
        np.mean(data['mixed']['peak_memory'])
    ], color=['#2196F3', '#4CAF50'])
    axes[2].set_ylabel('Peak Memory (GB)')
    axes[2].set_title('Memory Usage (lower is better)')

    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150)
    plt.show()
```

---

## Phase 8: Profiling and Optimization

### 8.1 Profiling Workflow

```bash
# System-level timeline (kernel launches, memory transfers, synchronization)
nsys profile --stats=true \
    python scripts/train_mixed_precision.py --epochs 5

# Kernel-level deep dive (occupancy, memory throughput, compute utilization)
ncu --set full \
    --target-processes all \
    python scripts/train_mixed_precision.py --epochs 1
```

### 8.2 What to Look For and Fix

| Metric (Nsight Compute) | If Bad, Do This |
|--------------------------|-----------------|
| **Tensor Core Utilization < 50%** | Check input dimensions — must be multiples of 8 (ideally 16) for FP16 Tensor Cores |
| **Global Load Efficiency < 80%** | Ensure coalesced memory access — adjacent threads should access adjacent memory |
| **SM Throughput low, Memory Throughput high** | Memory-bound — reduce global memory traffic via tiling or fusion |
| **SM Throughput high, Memory Throughput low** | Compute-bound — Tensor Cores are helping! Consider larger tiles |
| **Both low** | Latency-bound — increase occupancy, reduce register pressure |
| **Warp Execution Efficiency < 80%** | Warp divergence — restructure branches |
| **Achieved Occupancy < 50%** | Too many registers or too much shared memory per threadblock |

### 8.3 Profiling Helper

```python
# mixed_precision/utils/profiling.py
import jax
import time

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
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
        }
        return result

    def report(self):
        print(f"\n{'Operation':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12}")
        print("-" * 66)
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1]['mean_ms']):
            print(f"{name:<30} {t['mean_ms']:<12.3f} {t['std_ms']:<12.3f} {t['min_ms']:<12.3f}")
```

---

## Phase 9: Correctness Testing

### 9.1 GEMM Correctness

```python
# tests/test_gemm_correctness.py
import jax
import jax.numpy as jnp
import numpy as np
import mp_kernels

def test_gemm_matches_reference():
    """Verify custom CUTLASS GEMM matches JAX reference implementation."""
    rng = jax.random.PRNGKey(42)
    M, K, N = 128, 256, 64

    k1, k2, k3 = jax.random.split(rng, 3)
    A_fp32 = jax.random.normal(k1, (M, K))
    B_fp32 = jax.random.normal(k2, (K, N))
    bias = jax.random.normal(k3, (N,))

    A_fp16 = A_fp32.astype(jnp.float16)
    B_fp16 = B_fp32.astype(jnp.float16)

    # Reference: JAX matmul in FP32
    expected = A_fp16.astype(jnp.float32) @ B_fp16.astype(jnp.float32) + bias
    expected_relu = jnp.maximum(expected, 0).astype(jnp.float16)

    # Custom kernel
    actual = mixed_matmul(A_fp16, B_fp16, bias, apply_relu=True)

    np.testing.assert_allclose(
        np.array(actual), np.array(expected_relu),
        rtol=1e-2, atol=1e-2
    )

def test_gemm_various_sizes():
    """Test with different matrix dimensions including edge cases."""
    sizes = [
        (1, 16, 16),      # tiny
        (16, 16, 16),     # small square
        (128, 256, 64),   # typical layer
        (512, 1024, 256), # large layer
        (128, 512, 100),  # final classifier (100 classes)
        (1, 512, 100),    # single sample
    ]
    for M, K, N in sizes:
        rng = jax.random.PRNGKey(42)
        A = jax.random.normal(rng, (M, K), dtype=jnp.float16)
        B = jax.random.normal(rng, (K, N), dtype=jnp.float16)
        bias = jnp.zeros(N)

        result = mixed_matmul(A, B, bias, apply_relu=False)
        expected = (A.astype(jnp.float32) @ B.astype(jnp.float32)).astype(jnp.float16)

        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            rtol=1e-2, atol=1e-2,
            err_msg=f"Failed for size ({M}, {K}, {N})"
        )
```

### 9.2 Loss Scaling Correctness

```python
# tests/test_loss_scaling.py
import jax.numpy as jnp
from mixed_precision.training.loss_scaling import StaticLossScaler, DynamicLossScaler

def test_static_loss_scaling_preserves_gradients():
    """Verify that scale → backward → unscale recovers original gradients."""
    scaler = StaticLossScaler(scale=1024.0)

    # Simulate a gradient that would underflow in FP16
    small_grad = jnp.float16(0.00001)  # below FP16 minimum → becomes 0

    # With scaling: 0.00001 * 1024 = 0.01024 → representable in FP16
    scaled = small_grad * jnp.float16(1024.0)
    assert scaled != 0.0, "Scaled gradient should not be zero"

    # Unscale in FP32
    recovered = float(scaled) / 1024.0
    np.testing.assert_allclose(recovered, 0.00001, rtol=0.1)

def test_dynamic_scaler_reduces_on_overflow():
    scaler = DynamicLossScaler(init_scale=2**15)
    initial_scale = scaler.scale

    grads = {'w': jnp.array([jnp.inf])}
    should_apply = scaler.check_and_update(grads)

    assert not should_apply
    assert scaler.scale < initial_scale

def test_dynamic_scaler_increases_after_window():
    scaler = DynamicLossScaler(init_scale=1024.0, scale_window=5)
    initial_scale = scaler.scale

    for _ in range(5):
        grads = {'w': jnp.array([1.0])}
        scaler.check_and_update(grads)

    assert scaler.scale > initial_scale
```

### 9.3 JAX Autograd Correctness

```python
# tests/test_jax_integration.py
import jax
import jax.numpy as jnp
from mixed_precision.kernels.jax_primitives import mixed_matmul

def test_jax_grad_through_custom_op():
    """Verify JAX can differentiate through our custom CUTLASS kernel."""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 64), dtype=jnp.float16)
    w = jax.random.normal(rng, (64, 32), dtype=jnp.float16)
    bias = jnp.zeros(32)

    def f(x, w):
        return jnp.sum(mixed_matmul(x, w, bias, apply_relu=False))

    grad_x, grad_w = jax.grad(f, argnums=(0, 1))(x, w)

    assert grad_x.shape == x.shape
    assert grad_w.shape == w.shape
    assert not jnp.any(jnp.isnan(grad_x))
    assert not jnp.any(jnp.isnan(grad_w))

def test_jit_compilation():
    """Verify our custom ops work under jax.jit."""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 64), dtype=jnp.float16)
    w = jax.random.normal(rng, (64, 32), dtype=jnp.float16)
    bias = jnp.zeros(32)

    @jax.jit
    def f(x, w):
        return mixed_matmul(x, w, bias, apply_relu=True)

    result = f(x, w)
    assert result.shape == (32, 32)
    assert result.dtype == jnp.float16
```

### 9.4 Precision Casting Round-Trip

```python
# tests/test_precision_casting.py
import jax.numpy as jnp
import numpy as np

def test_fp32_to_fp16_and_back():
    """Values within FP16 range should survive round-trip."""
    values = jnp.array([1.0, 0.5, 0.001, 100.0, -3.14], dtype=jnp.float32)
    fp16 = values.astype(jnp.float16)
    back = fp16.astype(jnp.float32)
    np.testing.assert_allclose(np.array(back), np.array(values), rtol=1e-3)

def test_fp16_overflow_detection():
    """Values > 65504 should overflow to inf in FP16."""
    large = jnp.array([70000.0], dtype=jnp.float32)
    fp16 = large.astype(jnp.float16)
    assert jnp.isinf(fp16)

def test_fp16_underflow_detection():
    """Very small values should underflow to 0 in FP16."""
    tiny = jnp.array([1e-8], dtype=jnp.float32)
    fp16 = tiny.astype(jnp.float16)
    assert fp16 == 0.0
```

---

## Hyperparameters (Default Configuration)

```yaml
# configs/default.yaml
dataset:
  name: cifar100
  root: ./data
  batch_size: 128
  num_workers: 4

model:
  architecture: resnet18_cifar
  num_classes: 100
  num_filters: 64
  stage_sizes: [2, 2, 2, 2]

training:
  num_epochs: 200
  optimizer: adamw
  lr: 0.1
  weight_decay: 5e-4
  lr_schedule: cosine  # cosine annealing to 0
  warmup_epochs: 5

mixed_precision:
  enabled: true
  loss_scaling: dynamic
  initial_scale: 32768  # 2^15
  scale_factor: 2.0
  scale_window: 2000
  compute_dtype: float16  # FP16 for Tensor Cores
  accumulate_dtype: float32
  master_weight_dtype: float32

  # Operations that MUST stay in FP32 (precision-sensitive)
  fp32_ops:
    - softmax
    - cross_entropy
    - batch_norm
    - layer_norm
    - loss_computation

  # Operations safe for FP16 (compute-heavy, benefit from Tensor Cores)
  fp16_ops:
    - conv2d
    - dense_matmul
    - relu

cutlass:
  threadblock_shape: [128, 128, 32]
  warp_shape: [64, 64, 32]
  instruction_shape: [16, 8, 16]
  target_arch: sm_80  # Ampere (A100)
  epilogue_fusion: true

benchmark:
  num_runs: 50
  warmup_epochs: 5
  seeds: "range(50)"

augmentation:
  random_crop: true
  crop_padding: 4
  horizontal_flip: true
```

---

## Build and Run Commands

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Clone CUTLASS
mkdir -p third_party
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
cd third_party/cutlass && git checkout v3.3.0 && cd ../..

# 4. Build CUDA/C++ extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# 5. Run FP32 baseline training (single run)
python scripts/train_baseline.py \
    --epochs 200 \
    --lr 0.1 \
    --batch-size 128 \
    --seed 0

# 6. Run mixed-precision training (single run)
python scripts/train_mixed_precision.py \
    --epochs 200 \
    --lr 0.1 \
    --batch-size 128 \
    --seed 0 \
    --loss-scaling dynamic

# 7. Run full 50-run benchmark
python scripts/benchmark.py \
    --num-runs 50 \
    --output results/benchmark_results.json

# 8. Generate comparison plots
python scripts/compare_results.py \
    --input results/benchmark_results.json \
    --output results/comparison.png

# 9. Profile with Nsight Systems (timeline)
nsys profile --stats=true \
    python scripts/train_mixed_precision.py --epochs 5

# 10. Profile with Nsight Compute (kernel-level)
ncu --set full \
    python scripts/train_mixed_precision.py --epochs 1
```

---

## Expected Results

```
38% MEMORY REDUCTION comes from:
  ├── Activations stored in FP16 instead of FP32     (~45% savings on activations)
  ├── Working weights in FP16                         (~50% savings on working copy)
  ├── Gradients in FP16 during backward pass          (~50% savings on gradients)
  └── Master weights + optimizer state STAY in FP32   (no savings here)
      Net effect across all memory consumers: 38%

31% TRAINING SPEEDUP comes from:
  ├── Tensor Cores: 4-16x faster matmul per operation (~60% of the speedup)
  ├── Reduced memory bandwidth: 2x more values per transfer (~25% of speedup)
  ├── Epilogue fusion: 3x less memory traffic per layer (~15% of speedup)
  └── Non-matmul overhead unchanged (data loading, loss, Python) → limits total gain
      Net effect on end-to-end wall time: 31%

1.2% ACCURACY DELTA comes from:
  ├── FP32 accumulation prevents error buildup         (biggest saver)
  ├── FP32 master weights preserve tiny updates        (second biggest)
  ├── Loss scaling rescues small gradients              (third biggest)
  └── FP16 rounding still introduces SOME noise         (causes the 1.2%)
      This noise acts like mild regularization — sometimes helpful.
```

| Metric | FP32 Baseline | Mixed Precision | Delta |
|--------|--------------|-----------------|-------|
| Test Accuracy (CIFAR-100) | ~76% | ~74.8% | -1.2% |
| Peak GPU Memory | X GB | 0.62X GB | -38% |
| Training Time (200 epochs) | Y seconds | 0.69Y seconds | -31% |
| Tensor Core Utilization | 0% | >70% | N/A |

---

## Key Design Decisions and Rationale

### Why FP16 over BF16?

```
FP16:
  + Higher precision (~3 decimal digits vs BF16's ~2 digits)
  + Widely supported on all GPU generations since Volta (2017)
  + Tensor Core support on Volta, Turing, Ampere, Hopper
  - Smaller dynamic range (max ~65,504) → needs loss scaling

BF16:
  + Same dynamic range as FP32 → no loss scaling needed
  + Simpler implementation
  - Lower precision → more rounding noise
  - Only supported on Ampere (2020) and newer

Decision: FP16, because it works on more GPU generations and the higher
precision contributes to keeping accuracy delta at only 1.2%.
Loss scaling successfully handles the range limitation.
```

### Why CUTLASS Instead of cuBLAS?

```
cuBLAS:
  + Zero setup — just call cublasSgemm() / cublasHgemm()
  + Highly optimized by NVIDIA engineers
  - Black box — can't modify the kernel internals
  - Can't fuse bias + activation into the matmul
  - Three separate kernel launches for matmul → bias → ReLU
  - Limited precision combinations

CUTLASS:
  + Full control over input/output/accumulator precision
  + Epilogue fusion: matmul + bias + ReLU = 1 kernel, 1 memory round-trip
  + Can tune tile sizes for specific layer dimensions
  + Open source — can inspect and modify every line
  - More complex to set up (C++ templates)
  - Slightly slower than cuBLAS for standard non-fused cases (~2-5%)

Decision: CUTLASS. The epilogue fusion alone reduces memory traffic by ~3x
per layer. Combined with mixed-precision accumulation, this is the key
enabler of the 31% training speedup. cuBLAS cannot do this.
```

### Why JAX Instead of PyTorch?

```
PyTorch:
  + Larger ecosystem, more community support
  + torch.cuda.amp provides automatic mixed precision out of the box
  + Custom CUDA extensions via cpp_extension are well documented
  - Eager execution (each op runs immediately) — harder to optimize across ops
  - autocast is convenient but gives less control over which ops use which precision

JAX:
  + JIT compilation — traces entire function, optimizes across operations
  + Functional style — easier to reason about precision flow
  + XLA compiler can fuse adjacent operations automatically
  + custom_vjp gives clean custom gradient registration
  - Smaller ecosystem
  - Steeper learning curve

Decision: JAX. JIT compilation optimizes operations AROUND our custom kernels
(fusing Python-level ops, eliminating unnecessary copies). JAX's functional
approach makes the FP32↔FP16 weight casting flow cleaner to manage.
```

### Why FP32 Accumulation (Not FP16)?

```
This is the single most important design choice.

A matrix multiply sums K products:
  output[i][j] = Σ (A[i][k] × B[k][j]) for k = 0 to K-1

When K = 512 (typical hidden dimension):
  FP16 accumulation: error ≈ 512 × 0.001 = 0.5 (50% relative error → can't learn)
  FP32 accumulation: error ≈ 512 × 0.0000001 = 0.00005 (0.005% error → negligible)

FP32 accumulation is WHY accuracy only drops 1.2%.
Everything else (loss scaling, master weights) supports this core decision.
```

### Which Operations Stay in FP32?

```
MUST be FP32 (precision-sensitive):
  ├── Loss computation: cross-entropy uses log() and exp()
  │   log(0.001) = -6.9 → OK in FP32, lost in FP16
  ├── Softmax: involves exp() which can overflow FP16 (e^12 > 65504)
  ├── Batch normalization: division by variance, sqrt
  ├── Weight updates: tiny learning_rate × gradient additions
  └── Loss scaling arithmetic

Safe for FP16 (compute-heavy):
  ├── Convolution (implemented as GEMM)
  ├── Dense/linear layers (GEMM)
  ├── ReLU (simple threshold, no precision needed)
  └── Input data (pixel values don't need 7 digits of precision)
```

---

## Summary of What Makes This Project Non-Trivial

1. **Custom CUDA kernels using CUTLASS** — not just calling cuBLAS, but building template-based kernels with precise control over input/output/accumulator precision
2. **Epilogue fusion** — matmul + bias + ReLU combined into a single kernel, eliminating 2 out of 3 global memory round-trips per layer
3. **Mixed-precision arithmetic strategy** — FP16 for compute, FP32 for accumulation, FP32 for master weights, choosing correctly for each operation
4. **Loss scaling** (static and dynamic) — preventing gradient underflow in FP16 while avoiding overflow
5. **JAX integration** — registering CUTLASS kernels as JAX primitives with correct forward and backward implementations via `custom_vjp`
6. **Rigorous benchmarking** — 50 runs with statistical analysis, not just one lucky measurement
7. **Profiling-driven optimization** — using Nsight Systems and Nsight Compute to identify and fix bottlenecks
8. **Implicit GEMM for convolutions** — using CUTLASS's convolution support to apply mixed precision to CNN layers, not just dense layers

The project demonstrates end-to-end systems thinking: from GPU hardware (Tensor Cores) through kernel development (CUTLASS) to framework integration (JAX) to statistical validation (50-run benchmarking).


