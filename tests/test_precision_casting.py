"""Verify FP32 <-> FP16 round-trip behavior."""
import jax.numpy as jnp
import numpy as np


def test_fp32_to_fp16_and_back():
    values = jnp.array([1.0, 0.5, 0.001, 100.0, -3.14], dtype=jnp.float32)
    fp16 = values.astype(jnp.float16)
    back = fp16.astype(jnp.float32)
    np.testing.assert_allclose(np.array(back), np.array(values), rtol=1e-3)


def test_fp16_overflow_detection():
    large = jnp.array([70000.0], dtype=jnp.float32)
    fp16 = large.astype(jnp.float16)
    assert jnp.isinf(fp16)


def test_fp16_underflow_detection():
    tiny = jnp.array([1e-8], dtype=jnp.float32)
    fp16 = tiny.astype(jnp.float16)
    assert fp16 == 0.0
