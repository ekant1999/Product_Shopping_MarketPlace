cmake_minimum_required(VERSION 3.18)
project(mixed_precision_kernels CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# CUTLASS — clone or set path
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass")
if(EXISTS "${CUTLASS_DIR}/include")
  include_directories(${CUTLASS_DIR}/include)
  include_directories(${CUTLASS_DIR}/tools/util/include)
endif()

# CUDA architectures (Ampere 80, Ada 89, etc.)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

# Build shared library for Python binding
pybind11_add_module(mp_kernels
    csrc/kernels/mixed_precision_gemm.cu
    csrc/kernels/fused_linear_relu.cu
    csrc/kernels/fused_conv2d.cu
    csrc/loss_scaling/dynamic_loss_scaler.cu
    csrc/utils/precision_cast.cu
    csrc/bindings.cpp
)

target_include_directories(mp_kernels PRIVATE ${CMAKE_SOURCE_DIR}/csrc)

target_compile_options(mp_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)

target_link_libraries(mp_kernels PRIVATE ${CUDA_LIBRARIES} cublas)
