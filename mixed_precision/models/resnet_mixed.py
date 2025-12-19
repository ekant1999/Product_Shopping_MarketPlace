# mixed_precision/models/resnet_mixed.py
import jax.numpy as jnp
import flax.linen as nn


class MixedPrecisionResidualBlock(nn.Module):
    """ResNet block with FP16 conv params; BatchNorm and residual in FP32."""

    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x_fp16 = x.astype(jnp.float16)

        y = nn.Conv(
            self.filters,
            (3, 3),
            strides=(self.strides, self.strides),
            padding="SAME",
            use_bias=False,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
        )(x_fp16)
        y = y.astype(jnp.float32)
        y = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(y)
        y = nn.relu(y)

        y_fp16 = y.astype(jnp.float16)
        y = nn.Conv(
            self.filters,
            (3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
        )(y_fp16)
        y = y.astype(jnp.float32)
        y = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(y)

        if residual.shape != y.shape:
            residual_fp16 = residual.astype(jnp.float16)
            residual = nn.Conv(
                self.filters,
                (1, 1),
                strides=(self.strides, self.strides),
                use_bias=False,
                dtype=jnp.float16,
                param_dtype=jnp.float16,
            )(residual_fp16)
            residual = residual.astype(jnp.float32)
            residual = nn.BatchNorm(use_running_average=not train, dtype=jnp.float32)(
                residual
            )

        return nn.relu(y + residual)


class MixedPrecisionResNet(nn.Module):
    """ResNet with mixed-precision (FP16 conv, FP32 BN and classifier) for CIFAR-100."""

    stage_sizes: tuple = (2, 2, 2, 2)
    num_filters: int = 64
    num_classes: int = 100

    @nn.compact
    def __call__(self, x, train: bool = True):
        x_fp16 = x.astype(jnp.float16)

        x = nn.Conv(
            self.num_filters,
            (3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
        )(x_fp16)
        x = x.astype(jnp.float32)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for i, num_blocks in enumerate(self.stage_sizes):
            for j in range(num_blocks):
                strides = 2 if (i > 0 and j == 0) else 1
                filters = self.num_filters * (2**i)
                x = MixedPrecisionResidualBlock(filters=filters, strides=strides)(
                    x, train=train
                )

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=jnp.float32)(x)
        return x
