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
