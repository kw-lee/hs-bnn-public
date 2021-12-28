import jax.numpy as jnp
import jax.random as jr

def classification_data(seed=0):
    """
    Load 2D data. 2 Classes. Class labels generated from a 2-2-1 network.
    :param seed: random number seed
    :return:
    """
    key = jr.PRNGKey(seed)
    data = jnp.load("./data/2D_toy_data_linear.npz")
    x = data['x']
    y = data['y']
    ids = jnp.arange(x.shape[0])
    jr.shuffle(key, ids)
    # 75/25 split
    num_train = int(jnp.round(0.01*x.shape[0]))
    x_train = x[ids[:num_train]]
    y_train = y[ids[:num_train]]
    x_test = x[ids[num_train:]]
    y_test =y[ids[num_train:]]
    mu = jnp.mean(x_train, axis=0)
    std = jnp.std(x_train, axis=0)
    x_train = (x_train-mu)/std
    x_test = (x_test-mu)/std
    train_stats = dict()
    train_stats['mu'] = mu
    train_stats['sigma'] = std
    return x_train, y_train, x_test, y_test, train_stats


def regression_data(seed, data_count=500):
    """
    Generate data from a noisy sine wave.
    :param seed: random number seed
    :param data_count: number of data points.
    :return:
    """
    key = jr.PRNGKey(seed)
    noise_var = 0.1

    x = jnp.linspace(-4, 4, data_count)
    y = 1*jnp.sin(x) + jnp.sqrt(noise_var) * jr.normal(key, [data_count])

    train_count = int (0.2 * data_count)
    idx = jr.permutation(key, jnp.arange(data_count))
    x_train = x[idx[:train_count], jnp.newaxis ]
    x_test = x[ idx[train_count:], jnp.newaxis ]
    y_train = y[ idx[:train_count] ]
    y_test = y[ idx[train_count:] ]

    mu = jnp.mean(x_train, 0)
    std = jnp.std(x_train, 0)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
    mu = jnp.mean(y_train, 0)
    std = jnp.std(y_train, 0)
    y_train = (y_train - mu) / std
    train_stats = dict()
    train_stats['mu'] = mu
    train_stats['sigma'] = std

    return x_train, y_train, x_test, y_test, train_stats
