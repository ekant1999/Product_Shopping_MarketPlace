// csrc/kernels/mixed_precision_gemm.cu
// Mixed-precision GEMM: FP16 in, FP32 accum (via cuBLAS/custom), FP16 out.
// With CUTLASS: define CUTLASS_AVAILABLE and set CUTLASS_DIR for fused bias+ReLU.

#include "mixed_precision_gemm.h"
#include "utils/cuda_utils.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

__global__ void add_bias_relu_kernel(
    half* C, const float* bias, int M, int N, bool apply_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    float v = __half2float(C[idx]);
    if (bias) v += bias[col];
    if (apply_relu && v < 0.0f) v = 0.0f;
    C[idx] = __float2half(v);
}

#ifdef CUTLASS_AVAILABLE
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination.h>

using EpilogueWithRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t, 8, float, float
>;
using EpilogueNoRelu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 8, float, float
>;
using GemmWithRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueWithRelu
>;
using GemmNoRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueNoRelu
>;
#endif

void launch_mixed_precision_gemm(
    const half* A, const half* B, half* C, float* C_fp32,
    const float* bias, int M, int K, int N,
    bool apply_relu, cudaStream_t stream
) {
    (void)C_fp32;

#ifdef CUTLASS_AVAILABLE
    float alpha = 1.0f;
    float beta = 0.0f;
    if (apply_relu) {
        GemmWithRelu gemm_op;
        typename GemmWithRelu::Arguments args(
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
        if (bias) {
            int total = M * N;
            add_bias_relu_kernel<<<div_ceil(total, 256), 256, 0, stream>>>(
                C, bias, M, N, true);
            CUDA_CHECK_LAST_ERROR();
        }
    } else {
        GemmNoRelu gemm_op;
        typename GemmNoRelu::Arguments args(
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
        if (bias) {
            int total = M * N;
            add_bias_relu_kernel<<<div_ceil(total, 256), 256, 0, stream>>>(
                C, bias, M, N, false);
            CUDA_CHECK_LAST_ERROR();
        }
    }
#else
    // Fallback: cuBLAS HGEMM then add bias + ReLU
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);
    // C = A * B  with A [M,K], B [K,N], C [M,N] (row-major).
    // cuBLAS is column-major: (A*B)^T = B^T * A^T, so use OP_T, OP_T.
    cublasStatus_t st = cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        &alpha_h,
        reinterpret_cast<const __half*>(B), N,
        reinterpret_cast<const __half*>(A), K,
        &beta_h,
        reinterpret_cast<__half*>(C), N
    );
    cublasDestroy(handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Hgemm failed: %d\n", (int)st);
        exit(EXIT_FAILURE);
    }
    if (bias) {
        int total = M * N;
        add_bias_relu_kernel<<<div_ceil(total, 256), 256, 0, stream>>>(
            C, bias, M, N, apply_relu);
        CUDA_CHECK_LAST_ERROR();
    } else if (apply_relu) {
        int total = M * N;
        add_bias_relu_kernel<<<div_ceil(total, 256), 256, 0, stream>>>(
            C, nullptr, M, N, true);
        CUDA_CHECK_LAST_ERROR();
    }
#endif
}

