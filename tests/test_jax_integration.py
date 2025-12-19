"""Verify JAX autograd through custom ops."""
import jax
import jax.numpy as jnp
from mixed_precision.kernels.jax_primitives import mixed_matmul


def test_jax_grad_through_custom_op():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 64), dtype=jnp.float16)
    rng, _ = jax.random.split(rng)
    w = jax.random.normal(rng, (64, 32), dtype=jnp.float16)
    bias = jnp.zeros(32, dtype=jnp.float32)

    def f(x, w):
        return jnp.sum(mixed_matmul(x, w, bias, apply_relu=False))

    grad_x, grad_w = jax.grad(f, argnums=(0, 1))(x, w)
    assert grad_x.shape == x.shape
    assert grad_w.shape == w.shape
    assert not jnp.any(jnp.isnan(grad_x))
    assert not jnp.any(jnp.isnan(grad_w))


def test_jit_compilation():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 64), dtype=jnp.float16)
    w = jax.random.normal(rng, (64, 32), dtype=jnp.float16)
    bias = jnp.zeros(32, dtype=jnp.float32)

    @jax.jit
    def f(x, w):
        return mixed_matmul(x, w, bias, apply_relu=True)

    result = f(x, w)
    assert result.shape == (32, 32)
    assert result.dtype == jnp.float16
