import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from flow_matching.model.base import Model


class MLP(Model):
    dims: list[int]

    @nn.compact
    def forward(
        self, x: Float[ArrayLike, "*dims"], t: Float[ArrayLike, ""]
    ) -> Float[Array, "*dims"]:
        dims = jnp.shape(x)
        x = jnp.reshape(x, (-1))
        t = jnp.expand_dims(t, axis=0)

        for dim in self.dims[1:]:
            x = nn.Dense(dim)(x) + nn.Dense(dim, use_bias=False)(t)
            x = jax.nn.relu(x)

        x = nn.Dense(math.prod(dims))(x)
        x = jnp.reshape(x, dims)
        return x
