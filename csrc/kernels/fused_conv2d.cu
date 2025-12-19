// csrc/kernels/fused_conv2d.cu
// Placeholder: full impl uses CUTLASS implicit GEMM convolution.
// For now we provide a stub that returns without writing (callers should use JAX conv).
#include "fused_conv2d.h"
#include "utils/cuda_utils.h"

#ifdef CUTLASS_CONV_AVAILABLE
// When CUTLASS conv is available, include and call it here.
#else
void launch_fused_conv2d(
    const half* input, const half* filter, half* output,
    const float* bias,
    int N, int H, int W, int C_in,
    int C_out, int kH, int kW,
    int stride, int padding,
    bool apply_relu, cudaStream_t stream
) {
    (void)input;
    (void)filter;
    (void)output;
    (void)bias;
    (void)N;
    (void)H;
    (void)W;
    (void)C_in;
    (void)C_out;
    (void)kH;
    (void)kW;
    (void)stride;
    (void)padding;
    (void)apply_relu;
    (void)stream;
    // Stub: real implementation would use CUTLASS ImplicitGemmConvolution.
    // Training uses Flax nn.Conv in FP16; this is for future custom conv integration.
}
#endif
