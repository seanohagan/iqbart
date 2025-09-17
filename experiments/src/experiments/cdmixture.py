import numpy as np
from typing import Tuple

def p_func(x):
    """Probability of the normal component as a function of x"""
    return np.exp(-10 * (x - 0.8) ** 2)

def m_func(x):
    """Mean of the normal component as a function of x"""
    return 1 + 2 * (x - 0.8)

def gamma_shape_func(x):
    """Shape parameter for the gamma component as a function of x"""
    return 0.5 + x ** 2

def mu0_func(x):
    """Overall shift of the mixture as a function of x"""
    exp_term = np.exp(15 * (x - 0.5))
    return 5 * exp_term / (1 + exp_term) - 4 * x

def _generate_conditional_samples(x, n_samples=1000000, rng=None):
    """Generate samples from the mixture distribution at point x."""

    x_array = np.atleast_1d(x)
    results = []

    for x_val in x_array.flatten():
        z = rng.binomial(1, p_func(x_val), n_samples)
        normal_samples = rng.normal(m_func(x_val), 0.3, n_samples)
        gamma_samples = np.log(rng.gamma(gamma_shape_func(x_val), 1, n_samples))
        samples = z * normal_samples + (1 - z) * gamma_samples + mu0_func(x_val)
        results.append(samples)

    if len(x_array) == 1:
        return results[0]
    return results

def generate_data(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data from the covariate-dependent mixture model."""
    rng = np.random.RandomState(seed)

    X = rng.uniform(0, 1, n_samples)
    y = np.zeros_like(X)

    for i, x_val in enumerate(X):
        y[i] = _generate_conditional_samples(x_val, n_samples=1, rng=rng)

    return X.reshape(-1, 1), y

def compute_true_quantiles(x_grid: np.ndarray, q_grid: np.ndarray, seed: int) -> np.ndarray:
    """Compute true conditional quantiles via simulation."""
    x_flat = x_grid.flatten()
    results = np.zeros((len(x_flat), len(q_grid)))

    rng = np.random.RandomState(seed)

    n_sim_samples = 1000000

    for i, x_val in enumerate(x_flat):
        samples = _generate_conditional_samples(x_val, n_samples=n_sim_samples, rng=rng)
        results[i, :] = np.quantile(samples, q_grid)

    return results
