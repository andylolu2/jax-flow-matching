import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from flow_matching.model.base import Model


class MLP(Model):
    dims: list[int]

    @nn.compact
    def forward(
        self, x: Float[ArrayLike, "batch *dims"], t: Float[ArrayLike, "batch"]
    ) -> Float[ArrayLike, "batch *dims"]:
        batch, *dims = jnp.shape(x)

        x = nn.Dense(self.dims[0])(jnp.reshape(x, (batch, -1)))
        t = nn.Dense(self.dims[0])(jnp.reshape(t, (batch, 1)))
        x = jax.nn.relu(x + t)

        for dim in self.dims[1:]:
            x = nn.Dense(dim)(x)
            x = jax.nn.relu(x)

        x = nn.Dense(math.prod(dims))(x)
        x = jnp.reshape(x, (batch, *dims))
        return x
