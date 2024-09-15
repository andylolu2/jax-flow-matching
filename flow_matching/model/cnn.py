import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from flow_matching.model.base import Model


class CNN(Model):
    dims: list[int]

    @nn.compact
    def forward(
        self,
        x: Float[ArrayLike, "batch height width channels"],
        t: Float[ArrayLike, "batch"],
    ) -> Float[ArrayLike, "batch height width channels"]:
        b, h, w, c = x.shape
        for dim in self.dims:
            x = nn.Conv(dim, (3, 3), padding="SAME")(x)
            x += jnp.expand_dims(
                nn.Dense(dim)(jnp.expand_dims(t, axis=-1)), axis=(1, 2)
            )
            x = nn.relu(x)

        x = nn.Conv(c, (1, 1), padding="SAME")(x)
        return x
