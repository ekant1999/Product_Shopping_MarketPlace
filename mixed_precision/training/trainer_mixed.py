# mixed_precision/training/trainer_mixed.py
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .loss_scaling import DynamicLossScaler
from .trainer_fp32 import cross_entropy_loss
from ..data.cifar100 import make_batches


def create_mixed_train_state(rng, model, learning_rate, weight_decay):
    dummy_input = jnp.ones([1, 32, 32, 3])
    variables = model.init(rng, dummy_input, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    class MixedTrainState(train_state.TrainState):
        batch_stats: dict
        master_params: dict

    return MixedTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        master_params=jax.tree_util.tree_map(
            lambda p: p.astype(jnp.float32), params
        ),
    )


def train_step_mixed(state, images, labels, loss_scaler):
    fp16_params = jax.tree_util.tree_map(
        lambda p: p.astype(jnp.float16), state.master_params
    )

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updates = state.apply_fn(
            variables,
            images.astype(jnp.float16),
            train=True,
            mutable=["batch_stats"],
        )
        logits_fp32 = logits.astype(jnp.float32)
        loss = cross_entropy_loss(logits_fp32, labels)
        scaled_loss = loss_scaler.scale_loss(loss)
        return scaled_loss, (loss, logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (scaled_loss, (loss, logits, updates)), scaled_grads = grad_fn(fp16_params)

    grads_fp32 = loss_scaler.unscale_grads(scaled_grads)

    if not loss_scaler.check_and_update(grads_fp32):
        return state, loss, jnp.float32(0.0)

    updates_opt, new_opt_state = state.tx.update(
        grads_fp32, state.opt_state, state.master_params
    )
    new_master_params = jax.tree_util.tree_map(
        lambda p, u: p + u, state.master_params, updates_opt
    )

    state = state.replace(
        step=state.step + 1,
        opt_state=new_opt_state,
        master_params=new_master_params,
        params=fp16_params,
        batch_stats=updates["batch_stats"],
    )

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy


@jax.jit
def eval_step_mixed(state, images, labels):
    variables = {
        "params": jax.tree_util.tree_map(
            lambda p: p.astype(jnp.float32), state.master_params
        ),
        "batch_stats": state.batch_stats,
    }
    logits = state.apply_fn(variables, images, train=False)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return accuracy
