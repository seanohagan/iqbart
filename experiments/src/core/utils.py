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