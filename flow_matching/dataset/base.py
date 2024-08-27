from typing import Self

import jax.numpy as jnp
from flax import struct
from jaxtyping import ArrayLike, PRNGKeyArray, Shaped

# @flax.struct.dataclass
# class DatasetState:
#     epoch: int
#     step: int
#     rng: PRNGKeyArray


# class Dataset:
#     # @property
#     # def dimensions(self) -> tuple[int, ...]:
#     #     raise NotImplementedError

#     def sample(
#         self, state: DatasetState, batch_size: int
#     ) -> tuple[Shaped[ArrayLike, "{batch_size} ..."], DatasetState]:
#         raise NotImplementedError


@struct.dataclass
class Dataset:
    epoch: int
    step: int
    rng: PRNGKeyArray

    def sample(
        self, batch_size: int
    ) -> tuple[Shaped[ArrayLike, "{batch_size} ..."], Self]:
        raise NotImplementedError
