"""Verify loss scaling preserves gradients."""
import jax.numpy as jnp
import numpy as np
from mixed_precision.training.loss_scaling import StaticLossScaler, DynamicLossScaler


def test_static_loss_scaling_preserves_gradients():
    scaler = StaticLossScaler(scale=1024.0)
    small_grad = np.float32(0.00001)
    scaled = small_grad * 1024.0
    recovered = scaled / 1024.0
    np.testing.assert_allclose(recovered, 0.00001, rtol=0.1)


def test_dynamic_scaler_reduces_on_overflow():
    scaler = DynamicLossScaler(init_scale=2**15)
    initial_scale = scaler.scale
    grads = {"w": jnp.array([jnp.inf])}
    should_apply = scaler.check_and_update(grads)
    assert not should_apply
    assert scaler.scale < initial_scale


def test_dynamic_scaler_increases_after_window():
    scaler = DynamicLossScaler(init_scale=1024.0, scale_window=5)
    initial_scale = scaler.scale
    for _ in range(5):
        grads = {"w": jnp.array([1.0])}
        scaler.check_and_update(grads)
    assert scaler.scale > initial_scale
