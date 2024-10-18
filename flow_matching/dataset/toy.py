import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Shaped
from typing_extensions import Self

from flow_matching.dataset.base import Dataset, DatasetConfig


class ToyConfig(DatasetConfig):
    seed: int


@struct.dataclass
class ToyDataset(Dataset):
    means: Float[Array, "k 2"]
    covariances: Float[Array, "k 2 2"]

    @classmethod
    def create(cls, config: ToyConfig) -> Self:
        return cls(
            epoch=0,
            step=0,
            rng=jax.random.PRNGKey(config.seed),
            means=jnp.array([[4.0, -2.0], [-4.0, 3.0]]),
            covariances=jnp.array(
                [
                    [[1.0, 0.7], [0.7, 1.0]],
                    [[1.0, 0.7], [0.7, 1.0]],
                ]
            ),
        )

    def sample(self, batch_size: int) -> tuple[Shaped[Array, "{batch_size} 2"], Self]:
        rng1, rng2, *rngs = jax.random.split(self.rng, len(self.means) + 2)
        samples = jnp.array(
            [
                jax.random.multivariate_normal(
                    rngs[i], self.means[i], self.covariances[i], (batch_size,)
                )
                for i in range(len(self.means))
            ]
        )
        idx = jax.random.randint(rng2, (batch_size,), 0, len(self.means))
        samples = samples[idx, jnp.arange(batch_size)]
        return samples, self.replace(rng=rng1, step=self.step + 1)  # type: ignore


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    dataset = ToyDataset.create(ToyConfig(seed=0))
    samples, state = dataset.sample(1000)
    plt.scatter(samples[:, 0], samples[:, 1])

    Path("tmp").mkdir(exist_ok=True)
    plt.savefig("tmp/toy_dataset.png")
