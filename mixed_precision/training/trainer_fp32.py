# mixed_precision/training/trainer_fp32.py
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ..data.cifar100 import make_batches


def create_train_state(rng, model, learning_rate, weight_decay):
    dummy_input = jnp.ones([1, 32, 32, 3])
    variables = model.init(rng, dummy_input, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    class TrainState(train_state.TrainState):
        batch_stats: dict

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=100)
    return -jnp.mean(
        jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
    )


@jax.jit
def train_step_fp32(state, images, labels):
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updates = state.apply_fn(
            variables, images, train=True, mutable=["batch_stats"]
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy


@jax.jit
def eval_step(state, images, labels):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, images, train=False)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return accuracy


def train_fp32(model, train_data, test_data, config):
    rng = jax.random.PRNGKey(config["seed"])
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng, model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
    )

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    for epoch in range(config["num_epochs"]):
        rng, epoch_rng = jax.random.split(rng)
        for batch_imgs, batch_lbls in make_batches(
            train_images, train_labels, config["batch_size"], epoch_rng
        ):
            state, loss, acc = train_step_fp32(state, batch_imgs, batch_lbls)

        test_acc = eval_step(state, test_images, test_labels)

    return state, float(test_acc)
