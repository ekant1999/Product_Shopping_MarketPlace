# mixed_precision/models/resnet.py
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence


class ResidualBlock(nn.Module):
    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        y = nn.Conv(
            self.filters,
            (3, 3),
            strides=(self.strides, self.strides),
            padding="SAME",
            use_bias=False,
        )(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(
            self.filters, (3, 3), strides=(1, 1), padding="SAME", use_bias=False
        )(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(
                self.filters,
                (1, 1),
                strides=(self.strides, self.strides),
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """ResNet for CIFAR-100 (32x32 input)."""

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    num_filters: int = 64
    num_classes: int = 100

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(
            self.num_filters,
            (3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for i, num_blocks in enumerate(self.stage_sizes):
            for j in range(num_blocks):
                strides = 2 if (i > 0 and j == 0) else 1
                filters = self.num_filters * (2**i)
                x = ResidualBlock(filters=filters, strides=strides)(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
