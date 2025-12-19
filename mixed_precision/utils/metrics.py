# mixed_precision/utils/metrics.py
import jax.numpy as jnp


def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)
