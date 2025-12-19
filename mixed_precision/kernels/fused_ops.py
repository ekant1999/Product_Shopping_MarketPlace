# mixed_precision/kernels/fused_ops.py
"""Drop-in fused linear + bias + ReLU using custom GEMM."""
import jax
import jax.numpy as jnp
from .jax_primitives import mixed_matmul


def fused_linear_relu(x_fp16, w_fp16, bias_fp32, apply_relu=True):
    """Y = ReLU(X @ W + bias) in mixed precision."""
    return mixed_matmul(x_fp16, w_fp16, bias_fp32, apply_relu=apply_relu)
