// csrc/kernels/fused_linear_relu.cu
#include "fused_linear_relu.h"
#include "mixed_precision_gemm.h"

void launch_fused_linear_relu(
    const half* A, const half* B, const float* bias, half* C,
    int M, int K, int N,
    cudaStream_t stream
) {
    launch_mixed_precision_gemm(A, B, C, nullptr, bias, M, K, N, true, stream);
}
