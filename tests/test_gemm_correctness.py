"""Verify custom GEMM matches reference (within FP16 tolerance)."""
import numpy as np
import jax.numpy as jnp

try:
    import mp_kernels
    from mixed_precision.kernels.jax_primitives import mixed_matmul
    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False


def test_gemm_matches_reference():
    if not HAS_KERNEL:
        return
    rng = jax.random.PRNGKey(42)
    M, K, N = 128, 256, 64
    k1, k2, k3 = jax.random.split(rng, 3)
    A_fp32 = jax.random.normal(k1, (M, K))
    B_fp32 = jax.random.normal(k2, (K, N))
    bias = jax.random.normal(k3, (N,))

    A_fp16 = A_fp32.astype(jnp.float16)
    B_fp16 = B_fp32.astype(jnp.float16)
    expected = (
        A_fp16.astype(jnp.float32) @ B_fp16.astype(jnp.float32) + bias
    )
    expected_relu = jnp.maximum(expected, 0).astype(jnp.float16)

    actual = mixed_matmul(A_fp16, B_fp16, bias, apply_relu=True)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected_relu), rtol=1e-2, atol=1e-2
    )


def test_gemm_various_sizes():
    if not HAS_KERNEL:
        return
    sizes = [(16, 16, 16), (128, 256, 64), (128, 512, 100)]
    for M, K, N in sizes:
        rng = jax.random.PRNGKey(42)
        A = jax.random.normal(rng, (M, K), dtype=jnp.float16)
        rng, _ = jax.random.split(rng)
        B = jax.random.normal(rng, (K, N), dtype=jnp.float16)
        bias = jnp.zeros(N, dtype=jnp.float32)
        result = mixed_matmul(A, B, bias, apply_relu=False)
        expected = (A.astype(jnp.float32) @ B.astype(jnp.float32)).astype(
            jnp.float16
        )
        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            rtol=1e-2, atol=1e-2, err_msg=f"Failed for size ({M}, {K}, {N})"
        )
