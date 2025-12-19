// csrc/utils/precision_cast.cu
#include "precision_cast.h"
#include "cuda_utils.h"

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
