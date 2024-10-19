import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from typing_extensions import Self

from flow_matching.model.base import Model, ModelConfig


class CNNConfig(ModelConfig):
    dims: tuple[int, ...]


class CNN(Model):
    config: CNNConfig

    @classmethod
    def create(cls, config: CNNConfig) -> Self:
        return cls(config=config)

    @nn.compact
    def forward(
        self,
        x: Float[ArrayLike, "h w c"],
        t: Float[ArrayLike, ""],
        train: bool,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "h w c"]:
        h, w, c = jnp.shape(x)
        t = jnp.expand_dims(t, axis=0)
        for dim in self.config.dims:
            x = nn.Conv(dim, kernel_size=3)(x) + nn.Dense(dim, use_bias=False)(t)
            x = nn.relu(x)

        return nn.Conv(c, kernel_size=1)(x)
