// csrc/kernels/mixed_precision_gemm.h
#pragma once
#include <cuda_fp16.h>

void launch_mixed_precision_gemm(
    const half* A,           // [M x K] input activations in FP16
    const half* B,           // [K x N] weights in FP16
    half* C,                 // [M x N] output in FP16
    float* C_fp32,           // [M x N] output in FP32 (optional, for loss computation)
    const float* bias,       // [N] bias vector (FP32 for precision)
    int M, int K, int N,
    bool apply_relu,         // whether to fuse ReLU
    cudaStream_t stream = 0
);
