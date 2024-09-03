from flax import struct
from jaxtyping import Array, PRNGKeyArray, Shaped
from typing_extensions import Self


@struct.dataclass
class Dataset:
    epoch: int
    step: int
    rng: PRNGKeyArray

    def sample(self, batch_size: int) -> tuple[Shaped[Array, "{batch_size} ..."], Self]:
        raise NotImplementedError
