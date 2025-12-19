// csrc/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels/mixed_precision_gemm.h"
#include "kernels/fused_conv2d.h"
#include "utils/precision_cast.h"
#include <cuda_runtime.h>

namespace py = pybind11;

py::array_t<uint16_t> mixed_gemm(
    py::array_t<uint16_t> A,
    py::array_t<uint16_t> B,
    py::array_t<float> bias,
    bool apply_relu
) {
    auto buf_A = A.request();
    auto buf_B = B.request();
    auto buf_bias = bias.request();

    int M = static_cast<int>(buf_A.shape[0]);
    int K = static_cast<int>(buf_A.shape[1]);
    int N = static_cast<int>(buf_B.shape[1]);

    auto result = py::array_t<uint16_t>({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto buf_C = result.request();

    float* bias_ptr = (buf_bias.size > 0) ? static_cast<float*>(buf_bias.ptr) : nullptr;

    launch_mixed_precision_gemm(
        reinterpret_cast<const half*>(buf_A.ptr),
        reinterpret_cast<const half*>(buf_B.ptr),
        reinterpret_cast<half*>(buf_C.ptr),
        nullptr,
        bias_ptr,
        M, K, N,
        apply_relu,
        0
    );

    cudaDeviceSynchronize();
    return result;
}

PYBIND11_MODULE(mp_kernels, m) {
    m.doc() = "Mixed-precision CUDA kernels for deep learning";
    m.def("mixed_gemm", &mixed_gemm,
          "Mixed-precision GEMM (FP16 in, FP32 accum, FP16 out)",
          py::arg("A"), py::arg("B"), py::arg("bias"), py::arg("apply_relu") = true);
}
