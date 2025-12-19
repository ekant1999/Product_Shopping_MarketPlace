# mixed_precision/data/augmentation.py
import jax
import jax.numpy as jnp


def random_crop_and_flip(rng, image, padding=4):
    """Random crop with padding and horizontal flip for CIFAR."""
    rng_crop, rng_flip = jax.random.split(rng)
    padded = jnp.pad(
        image,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="reflect",
    )
    h_offset = jax.random.randint(rng_crop, (), 0, 2 * padding + 1)
    w_offset = jax.random.randint(rng_crop, (), 0, 2 * padding + 1)
    cropped = jax.lax.dynamic_slice(
        padded, (h_offset, w_offset, 0), (32, 32, 3)
    )
    flip = jax.random.bernoulli(rng_flip)
    cropped = jnp.where(flip, jnp.flip(cropped, axis=1), cropped)
    return cropped


def augment_batch(rng, images):
    """Apply augmentation to a batch of images."""
    batch_size = images.shape[0]
    rngs = jax.random.split(rng, batch_size)
    return jax.vmap(random_crop_and_flip)(rngs, images)
