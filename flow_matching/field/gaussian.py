import jax
import jax.numpy as jnp

def _u(x, t, x1, mu, sigma):
    """u_t(x | x_1) = sigma'_t(x_1) / sigma_t(x_1) (x - mu_t(x_1)) + mu'_t(x_1)
    
    mu: (x, t) -> R^d
    sigma: (x, t) -> R

    -(1 - 1e-5) / (1 - (1 - 1e-5) * t) * (x - t * x_1) + x_1
    = -(1 - 1e-5)
    """
    dmu_dt = jax.jacfwd(mu, argnums=1)
    dsigma_dt = jax.jacfwd(sigma, argnums=1)
    
    return (dsigma_dt(x1, t) / sigma(x1, t)) * (x - mu(x1, t)) + dmu_dt(x1, t)

def u_ot(x, t, x1):
    # mu = lambda x, t: t * x1
    # sigma = lambda x, t: 1 - (1 - 1e-5) * t
    # return _u(x, t, x1, mu, sigma)

    return (x1 - (1 - 1e-5)  *x) / (1 - (1 - 1e-5) * t)