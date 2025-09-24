import numpy as np
from typing import Tuple

def generate_grid(seed: int, domain: Tuple[float, float],
                 quantile_range: Tuple[float, float],
                 grid_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate evaluation grid for x and quantile values."""
    rng = np.random.RandomState(seed)

    x_min, x_max = domain
    q_min, q_max = quantile_range

    x_grid = rng.uniform(x_min, x_max, grid_size).reshape(-1, 1)
    q_grid = rng.uniform(q_min, q_max, grid_size)

    return x_grid, q_grid

def get_data_seed(repetition: int) -> int:
    """Get deterministic seed for data generation based on repetition."""
    return repetition

def get_grid_seed(repetition: int, seed_offset: int = 1000) -> int:
    """Get deterministic seed for grid generation based on repetition."""
    return repetition + seed_offset

def generate_fixed_grid(domain: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fixed evaluation grid for consistent metrics across experiments.

    Args:
        domain: (x_min, x_max) for the covariate domain

    Returns:
        x_fixed: 20 interior points uniformly spaced in domain
        q_fixed: (0.05, 0.1, 0.15, ..., 0.95) quantiles
    """
    x_min, x_max = domain

    # Generate 22 uniformly spaced points, exclude first and last to get 20 interior points
    x_interior = np.linspace(x_min, x_max, 22)[1:-1].reshape(-1, 1)

    # Fixed quantiles from 0.05 to 0.95 in steps of 0.05
    q_fixed = np.arange(0.05, 1.0, 0.05)

    return x_interior, q_fixed