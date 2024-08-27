import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray


def mu(x: Float[ArrayLike, "d"], t: Float[ArrayLike, ""]) -> Float[Array, "d"]:
    return jnp.array(t * x)


def sigma(x: Float[ArrayLike, "d"], t: Float[ArrayLike, ""]) -> Float[Array, ""]:
    return jnp.array(1 - (1 - 1e-5) * t)


def sample(
    t: Float[ArrayLike, ""], x1: Float[ArrayLike, "d"], rng: PRNGKeyArray
) -> Float[Array, "d"]:
    """Sample x ~ p_t(x | x_1)"""
    return jax.random.multivariate_normal(
        rng, mu(x1, t), sigma(x1, t) ** 2 * jnp.eye(jnp.shape(x1)[-1])
    )


def p(
    x: Float[ArrayLike, "d"], t: Float[ArrayLike, ""], x1: Float[ArrayLike, "d"]
) -> Float[Array, ""]:
    """p_t(x | x_1) = N(x | mu(x_1, t), sigma(x_1, t))"""
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mu(x1, t), sigma(x1, t) ** 2 * jnp.eye(jnp.shape(x)[-1])
    )


def u(
    x: Float[ArrayLike, "d"], t: Float[ArrayLike, ""], x1: Float[ArrayLike, "d"]
) -> Float[Array, "d"]:
    """u_t(x | x_1) = sigma'_t(x_1) / sigma_t(x_1) (x - mu_t(x_1)) + mu'_t(x_1)

    mu: (x, t) -> R^d
    sigma: (x, t) -> R
    """
    dmu_dt = jax.jacfwd(mu, argnums=1)
    dsigma_dt = jax.jacfwd(sigma, argnums=1)

    return (dsigma_dt(x1, t) / sigma(x1, t)) * (x - mu(x1, t)) + dmu_dt(x1, t)
