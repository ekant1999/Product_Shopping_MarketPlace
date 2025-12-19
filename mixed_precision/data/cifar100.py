# mixed_precision/data/cifar100.py
import jax
import tensorflow_datasets as tfds
import jax.numpy as jnp
import numpy as np

CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


def load_cifar100(batch_size=128):
    """Load CIFAR-100 train and test sets with per-channel normalization."""
    ds_train, ds_info = tfds.load("cifar100", split="train", with_info=True)
    ds_test = tfds.load("cifar100", split="test")

    def normalize(sample):
        image = np.array(sample["image"], dtype=np.float32) / 255.0
        image = (image - CIFAR100_MEAN) / CIFAR100_STD
        label = int(sample["label"])
        return image, label

    train_images, train_labels = [], []
    for sample in tfds.as_numpy(ds_train):
        img, lbl = normalize(sample)
        train_images.append(img)
        train_labels.append(lbl)

    test_images, test_labels = [], []
    for sample in tfds.as_numpy(ds_test):
        img, lbl = normalize(sample)
        test_images.append(img)
        test_labels.append(lbl)

    train_images = jnp.array(np.stack(train_images))
    train_labels = jnp.array(np.array(train_labels))
    test_images = jnp.array(np.stack(test_images))
    test_labels = jnp.array(np.array(test_labels))

    return (train_images, train_labels), (test_images, test_labels)


def make_batches(images, labels, batch_size, rng_key, shuffle=True):
    """Yield (images, labels) batches. rng_key used for shuffle."""
    n = images.shape[0]
    if shuffle:
        perm = jax.random.permutation(rng_key, n)
        images = images[perm]
        labels = labels[perm]

    num_batches = n // batch_size
    for i in range(num_batches):
        start = i * batch_size
        yield images[start : start + batch_size], labels[start : start + batch_size]


