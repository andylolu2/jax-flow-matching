from typing import Literal

import datasets
import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Shaped
from typing_extensions import Self

from flow_matching.dataset.base import Dataset, DatasetConfig


class Cifar10Config(DatasetConfig):
    seed: int
    split: Literal["train", "val"]


@struct.dataclass
class Cifar10Dataset(Dataset):
    img: Float[Array, "n 28 28"]

    @classmethod
    def create(cls, config: Cifar10Config) -> Self:
        ds = datasets.load_dataset("cifar10")
        assert isinstance(ds, datasets.DatasetDict)

        if config.split == "train":
            img = jnp.array(ds["train"]["img"][:40000])
        elif config.split == "val":
            img = jnp.array(ds["train"]["img"][40000:])
        else:
            raise ValueError(f"Unknown split: {config.split}")

        img = img.astype(jnp.float32) / 255.0  # Int[0, 255] -> Float[0, 1]

        return cls(epoch=0, step=0, rng=jax.random.PRNGKey(config.seed), img=img)

    def sample(
        self, batch_size: int
    ) -> tuple[Shaped[Array, "{batch_size} 32 32 3"], Self]:
        assert (
            0 < batch_size <= len(self.img)
        ), f"Invalid {batch_size=} but {self.img.shape=}"

        state = jax.lax.cond(
            self.step + batch_size > len(self.img),
            lambda: self.replace(epoch=self.epoch + 1, step=0),  # type: ignore
            lambda: self,
        )

        order = jax.random.permutation(state.rng, len(state.img))
        sample_idx = jax.lax.dynamic_slice_in_dim(order, state.step, batch_size, axis=0)
        return (
            state.img[sample_idx],
            state.replace(step=state.step + batch_size),  # type: ignore
        )
