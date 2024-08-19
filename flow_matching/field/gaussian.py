import jax
import jax.numpy as jnp


def mu(x, t):
    return t * x


def sigma(x, t):
    return 1 - (1 - 1e-5) * t


def p(x, t, x1):
    """p_t(x | x_1) = N(x | mu(x_1, t), sigma(x_1, t))"""
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mu(x1, t), sigma(x1, t) ** 2 * jnp.eye(x.shape[-1])
    )


def u(x, t, x1):
    """u_t(x | x_1) = sigma'_t(x_1) / sigma_t(x_1) (x - mu_t(x_1)) + mu'_t(x_1)

    mu: (x, t) -> R^d
    sigma: (x, t) -> R
    """
    dmu_dt = jax.jacfwd(mu, argnums=1)
    dsigma_dt = jax.jacfwd(sigma, argnums=1)

    return (dsigma_dt(x1, t) / sigma(x1, t)) * (x - mu(x1, t)) + dmu_dt(x1, t)
