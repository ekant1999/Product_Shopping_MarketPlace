// csrc/loss_scaling/dynamic_loss_scaler.cu
#include "dynamic_loss_scaler.h"
#include "../utils/cuda_utils.h"
#include <cuda_runtime.h>

__global__ void check_inf_nan_kernel(const float* g, int n, int* has_bad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = g[idx];
    if (isinf(v) || isnan(v)) *has_bad = 1;
}

void dynamic_loss_scale_step(
    const float* grads_fp32, int n,
    float* scale_inout, int* good_steps_inout,
    float scale_factor, int scale_window,
    cudaStream_t stream
) {
    (void)scale_factor;
    (void)scale_window;
    int* d_has_bad = nullptr;
    cudaMalloc(&d_has_bad, sizeof(int));
    cudaMemset(d_has_bad, 0, sizeof(int));
    check_inf_nan_kernel<<<div_ceil(n, 256), 256, 0, stream>>>(grads_fp32, n, d_has_bad);
    int has_bad = 0;
    cudaMemcpyAsync(&has_bad, d_has_bad, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_has_bad);
    if (has_bad) {
        *scale_inout *= 0.5f;
        *good_steps_inout = 0;
    } else {
        (*good_steps_inout)++;
        if (*good_steps_inout >= (int)scale_window) {
            *scale_inout *= 2.0f;
            *good_steps_inout = 0;
        }
    }
}
