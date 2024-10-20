import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from typing_extensions import Self

from flow_matching.model.base import Model, ModelConfig


class MLPConfig(ModelConfig):
    dims: tuple[int, ...]


class MLP(Model):
    config: MLPConfig

    @classmethod
    def create(cls, config: MLPConfig) -> Self:
        return cls(config=config)

    @nn.compact
    def forward(
        self,
        x: Float[ArrayLike, "*dims"],
        t: Float[ArrayLike, ""],
        train: bool,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "*dims"]:
        dims = jnp.shape(x)
        x = jnp.reshape(x, (-1))
        t = jnp.expand_dims(t, axis=0)

        for dim in self.config.dims[1:]:
            x = nn.Dense(dim)(x) + nn.Dense(dim, use_bias=False)(t)
            x = jax.nn.relu(x)

        x = nn.Dense(math.prod(dims))(x)
        x = jnp.reshape(x, dims)
        return x
