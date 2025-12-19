// csrc/kernels/fused_linear_relu.h
#pragma once
#include <cuda_fp16.h>

// Thin wrapper: fused linear (matmul + bias + ReLU) using mixed-precision GEMM
void launch_fused_linear_relu(
    const half* A,      // [M x K]
    const half* B,      // [K x N]
    const float* bias,   // [N]
    half* C,             // [M x N]
    int M, int K, int N,
    cudaStream_t stream = 0
);
