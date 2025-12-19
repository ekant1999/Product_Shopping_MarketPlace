# mixed_precision/training/loss_scaling.py
import jax
import jax.numpy as jnp


class StaticLossScaler:
    def __init__(self, scale: float = 1024.0):
        self.scale = scale

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, grads):
        return jax.tree_util.tree_map(
            lambda g: g.astype(jnp.float32) / self.scale, grads
        )


class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 2**15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
    ):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, grads):
        return jax.tree_util.tree_map(
            lambda g: g.astype(jnp.float32) / self.scale, grads
        )

    def check_and_update(self, grads):
        """Check for overflow and adjust scale. Returns True if step should be applied."""
        leaves = jax.tree_util.tree_leaves(grads)
        has_inf_or_nan = any(
            jnp.any(jnp.isinf(g) | jnp.isnan(g)) for g in leaves
        )

        if has_inf_or_nan:
            self.scale /= self.scale_factor
            self.good_steps = 0
            return False
        self.good_steps += 1
        if self.good_steps >= self.scale_window:
            self.scale *= self.scale_factor
            self.good_steps = 0
        return True
