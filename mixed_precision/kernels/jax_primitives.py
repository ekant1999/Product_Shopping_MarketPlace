# mixed_precision/kernels/jax_primitives.py
"""JAX custom_vjp registration for mixed-precision GEMM (mp_kernels.mixed_gemm)."""
import numpy as np
import jax
import jax.numpy as jnp

try:
    import mp_kernels
except ImportError:
    mp_kernels = None


def _mixed_matmul_impl(x_fp16, w_fp16, bias_fp32, apply_relu=True):
    """Forward pass using CUDA kernel if available, else JAX."""
    if mp_kernels is not None:
        # Ensure contiguous arrays for the kernel (JAX/NumPy)
        A = np.asarray(x_fp16.view(jnp.uint16) if hasattr(x_fp16, "view") else x_fp16, dtype=np.uint16)
        B = np.asarray(w_fp16.view(jnp.uint16) if hasattr(w_fp16, "view") else w_fp16, dtype=np.uint16)
        bias = np.asarray(bias_fp32, dtype=np.float32)
        result_uint16 = mp_kernels.mixed_gemm(A, B, bias, apply_relu)
        out = np.asarray(result_uint16).view(np.float16).reshape(
            x_fp16.shape[0], w_fp16.shape[1]
        )
        return jnp.array(out)
    # Fallback: pure JAX (no custom kernel)
    out = jnp.dot(x_fp16.astype(jnp.float32), w_fp16.astype(jnp.float32)) + bias_fp32
    if apply_relu:
        out = jnp.maximum(out, 0.0)
    return out.astype(jnp.float16)


def _mixed_matmul_fwd(x_fp16, w_fp16, bias_fp32, apply_relu=True):
    result = _mixed_matmul_impl(x_fp16, w_fp16, bias_fp32, apply_relu)
    return result, (x_fp16, w_fp16, result, apply_relu)


def _mixed_matmul_bwd(saved, grad_output):
    x, w, result, apply_relu = saved
    if apply_relu:
        grad_output = jnp.where(result > 0, grad_output, 0.0)
    grad_x = _mixed_matmul_impl(
        grad_output, w.T, jnp.zeros(x.shape[1], dtype=jnp.float32), apply_relu=False
    )
    grad_w = _mixed_matmul_impl(
        x.T, grad_output, jnp.zeros(w.shape[1], dtype=jnp.float32), apply_relu=False
    )
    grad_bias = jnp.sum(grad_output.astype(jnp.float32), axis=0)
    return grad_x, grad_w, grad_bias, None


if mp_kernels is not None:
    mixed_matmul = jax.custom_vjp(_mixed_matmul_impl)
    mixed_matmul.defvjp(_mixed_matmul_fwd, _mixed_matmul_bwd)
else:
    mixed_matmul = _mixed_matmul_impl
