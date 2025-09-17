import numpy as np
import scipy.stats as stats
from typing import Tuple

def _lstar_conditional_inverse_cdf(y_t_prev, q, rho1=0, rho2=0.9, gamma=5, c=0, sigma_eps=1, nu=3):
    """Compute conditional inverse CDF (quantile function) of Y_t given Y_{t-1}."""
    F = 1 / (1 + np.exp(-gamma * (y_t_prev - c)))
    conditional_mean = rho1 * y_t_prev + rho2 * F * y_t_prev
    epsilon_quantile = stats.t.ppf(q, df=nu)
    y_t_quantile = conditional_mean + sigma_eps * epsilon_quantile
    return y_t_quantile

def _simulate_lstar(rng, rho1=0, rho2=0.9, gamma=5, c=0, sigma_eps=1, nu=3, T=15000):
    """Simulate data from the LSTAR model."""
    Y = np.zeros(T)
    eps_t = stats.t.rvs(df=nu, size=T, random_state=rng)

    for t in range(1, T):
        F = 1 / (1 + np.exp(-gamma * (Y[t-1] - c)))
        Y[t] = rho1 * Y[t-1] + rho2 * F * Y[t-1] + sigma_eps * eps_t[t]

    return Y

def generate_data(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data from the LSTAR model."""
    rng = np.random.RandomState(seed)

    ts = _simulate_lstar(rng, rho1=0, rho2=0.9, gamma=5, c=0, sigma_eps=1, nu=3, T=15000)

    X = ts[:-1].reshape(-1, 1)
    y = ts[1:]

    if n_samples < len(X):
        X = X[:n_samples]
        y = y[:n_samples]

    return X, y

def compute_true_quantiles(x_grid: np.ndarray, q_grid: np.ndarray, seed: int) -> np.ndarray:
    """Compute true conditional quantiles using the analytical formula."""
    x_flat = x_grid.flatten()
    results = np.zeros((len(x_flat), len(q_grid)))

    for i, x_val in enumerate(x_flat):
        for j, q_val in enumerate(q_grid):
            results[i, j] = _lstar_conditional_inverse_cdf(x_val, q_val)

    return results
