# mixed_precision/kernels/custom_matmul.py
"""Python wrapper for CUTLASS/cuBLAS mixed-precision GEMM."""
from .jax_primitives import mixed_matmul

__all__ = ["mixed_matmul"]
