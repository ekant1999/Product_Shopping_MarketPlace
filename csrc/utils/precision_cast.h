// csrc/utils/precision_cast.h
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void cast_fp32_to_fp16(const float* input, half* output, int n, cudaStream_t stream = 0);
void cast_fp16_to_fp32(const half* input, float* output, int n, cudaStream_t stream = 0);
