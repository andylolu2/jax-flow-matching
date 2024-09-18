import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
from clu.metrics import Average, Collection
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from flow_matching.field import gaussian_flow

Loss = Float[Array, ""]


@flax.struct.dataclass
class ModelMetrics(Collection):
    loss: Average.from_output("loss")  # type: ignore


class Model(nn.Module):
    def __call__(
        self, x1: Float[ArrayLike, "batch *dims"], rng: PRNGKeyArray
    ) -> tuple[Loss, ModelMetrics]:
        batch = jnp.shape(x1)[0]
        t_rng, x_rng = jax.random.split(rng)

        t = jax.random.uniform(t_rng, (batch,))
        x = jax.vmap(gaussian_flow.sample)(t, x1, jax.random.split(x_rng, batch))

        u_target = jax.vmap(gaussian_flow.u)(x, t, x1)
        u_pred = jax.vmap(self.forward)(x, t)
        loss = jnp.mean(jnp.sum((u_pred - u_target) ** 2, axis=-1))
        return loss, ModelMetrics.single_from_model_output(loss=loss)

    def forward(
        self, x: Float[ArrayLike, "*dims"], t: Float[ArrayLike, ""]
    ) -> Float[Array, "*dims"]:
        raise NotImplementedError
