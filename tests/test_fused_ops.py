"""Verify fused linear+ReLU produces correct outputs."""
import numpy as np
import jax
import jax.numpy as jnp
from mixed_precision.kernels.fused_ops import fused_linear_relu


def test_fused_linear_relu_shape():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 64), dtype=jnp.float16)
    rng, _ = jax.random.split(rng)
    w = jax.random.normal(rng, (64, 32), dtype=jnp.float16)
    bias = jnp.zeros(32, dtype=jnp.float32)
    out = fused_linear_relu(x, w, bias, apply_relu=True)
    assert out.shape == (32, 32)
    assert out.dtype == jnp.float16


def test_fused_linear_relu_nonnegative():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (8, 16), dtype=jnp.float16)
    w = jax.random.normal(rng, (16, 8), dtype=jnp.float16)
    bias = jnp.zeros(8, dtype=jnp.float32)
    out = fused_linear_relu(x, w, bias, apply_relu=True)
    np.testing.assert_array_less(-1e-5, np.array(out))
