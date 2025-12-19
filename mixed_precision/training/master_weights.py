# mixed_precision/training/master_weights.py
import jax
import jax.numpy as jnp


class MasterWeightManager:
    def __init__(self, params):
        self.master_params = jax.tree_util.tree_map(
            lambda p: p.astype(jnp.float32), params
        )

    def get_fp16_params(self):
        return jax.tree_util.tree_map(
            lambda p: p.astype(jnp.float16), self.master_params
        )

    def update(self, grads_fp32, optimizer_state, tx):
        updates, new_opt_state = tx.update(
            grads_fp32, optimizer_state, self.master_params
        )
        self.master_params = jax.tree_util.tree_map(
            lambda p, u: p + u, self.master_params, updates
        )
        return new_opt_state

    def get_master_params(self):
        return self.master_params
