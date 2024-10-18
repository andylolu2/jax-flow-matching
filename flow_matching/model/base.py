import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
from clu.metrics import Average, Collection
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from pydantic import BaseModel

from flow_matching.field import gaussian_flow

Loss = Float[Array, ""]


class ModelConfig(BaseModel): ...


@flax.struct.dataclass
class ModelMetrics(Collection):
    loss: Average.from_output("loss")  # type: ignore


class Model(nn.Module):
    def __call__(
        self, x1: Float[ArrayLike, "*dims"], rng: PRNGKeyArray, train: bool
    ) -> tuple[Loss, ModelMetrics]:
        # batch = jnp.shape(x1)[0]
        t_rng, x_rng, fwd_rng = jax.random.split(rng, 3)

        t = jax.random.uniform(t_rng)
        x = gaussian_flow.sample(t, x1, x_rng)

        u_target = gaussian_flow.u(x, t, x1)
        u_pred = self.forward(x, t, train, fwd_rng)
        loss = jnp.mean(jnp.sum((u_pred - u_target) ** 2))
        return loss, ModelMetrics.single_from_model_output(loss=loss)

    def forward(
        self,
        x: Float[ArrayLike, "*dims"],
        t: Float[ArrayLike, ""],
        train: bool,
        rng: PRNGKeyArray | None = None,
    ) -> Float[Array, "*dims"]:
        raise NotImplementedError
