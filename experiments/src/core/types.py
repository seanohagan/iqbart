from typing import Callable, Tuple, NamedTuple, Union, Optional
import numpy as np

class PredictionResult(NamedTuple):
    predictions: np.ndarray  # (n_test_points, n_quantiles) - always present
    posterior_samples: Optional[np.ndarray] = None  # (n_chains, n_test_points, n_quantiles, n_draws) - for coverage metrics
    fixed_predictions: Optional[np.ndarray] = None # (n_fixed_points, n_fixed_quantiles)
    fixed_posterior_samples: Optional[np.ndarray] = None  # (n_chains, n_fixed_points, n_fixed_quantiles, n_draws) - for coverage metrics

Method = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Union[np.ndarray, PredictionResult]]

DataGenerator = Callable[[int, int], Tuple[np.ndarray, np.ndarray]]

QuantileComputer = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

class ExperimentDGP(NamedTuple):
    generate_data: DataGenerator
    compute_true_quantiles: QuantileComputer
