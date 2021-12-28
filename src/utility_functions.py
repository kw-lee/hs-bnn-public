import jax.numpy as jnp
from jax.scipy.special import gammaln, digamma
import seaborn as sb
import matplotlib.pyplot as plt


def diag_gaussian_entropy(log_std, D):
    return 0.5 * D * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(log_std)


def inv_gamma_entropy(a, b):
    return jnp.sum(a + jnp.log(b) + gammaln(a) - (1 + a) * digamma(a))


def log_normal_entropy(log_std, mu, D):
    return jnp.sum(log_std + mu + 0.5) + (D / 2) * jnp.log(2 * jnp.pi)


def make_batches(n_data, batch_size):
    return [slice(i, min(i+batch_size, n_data)) for i in range(0, n_data, batch_size)]


