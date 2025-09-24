import time
import numpy as np
from typing import Dict, Any, Optional
from .registry import get_method, get_experiment_dgp, validate_method_params, validate_experiment_params
from .utils import generate_grid, get_data_seed, get_grid_seed, generate_fixed_grid
from .types import PredictionResult
from ..evaluation.metrics import compute_all_wasserstein_metrics, compute_coverage_metrics, compute_pinball_metrics

def run_single_experiment(
    experiment_name: str,
    method_name: str,
    experiment_params: Dict[str, Any],
    method_params: Dict[str, Any],
    n_data: int,
    repetition: int,
    eval_grid_size: int = 100,
    seed_offset: int = 1000
) -> Dict[str, Any]:
    """
    Run a single experimental trial.

    Returns dictionary with metrics, timing, and exit status.
    """
    try:
        # Validate parameters
        exp_params = validate_experiment_params(experiment_name, experiment_params)
        meth_params = validate_method_params(method_name, method_params)

        # Get experiment DGP and method function
        experiment_dgp = get_experiment_dgp(experiment_name)
        method_fn = get_method(method_name)

        # Generate seeds
        data_seed = get_data_seed(repetition)
        grid_seed = get_grid_seed(repetition, seed_offset)
        method_seed = repetition + 2000  # Method seed offset
        quantile_estimation_seed = repetition + 3000

        # Generate data
        X, y = experiment_dgp.generate_data(n_data, data_seed)

        # Generate evaluation grid
        domain = exp_params.get('domain', [[0.0, 1.0]])[0]
        quantile_range = exp_params.get('quantile_range', [0.01, 0.99])
        x_grid, q_grid = generate_grid(grid_seed, domain, quantile_range, eval_grid_size)

        # Generate fixed evaluation grid for consistent metrics
        x_fixed, q_fixed = generate_fixed_grid(domain)

        # Run method with timing
        start_time = time.time()
        method_result = method_fn(X, y, x_grid, q_grid, seed=method_seed, X_test_fixed=x_fixed, q_fixed=q_fixed, **meth_params)
        fit_predict_time = time.time() - start_time

        # Handle Union return type - extract predictions and optional posterior samples
        if isinstance(method_result, PredictionResult):
            predictions = method_result.predictions
            posterior_samples = method_result.posterior_samples
            fixed_predictions = method_result.fixed_predictions
            fixed_posterior_samples = method_result.fixed_posterior_samples
        else:
            predictions = method_result  # Plain ndarray (backward compatibility)
            posterior_samples = None
            fixed_predictions = None
            fixed_posterior_samples = None

        # Compute true quantiles for random grid
        true_quantiles = experiment_dgp.compute_true_quantiles(x_grid, q_grid, quantile_estimation_seed)

        # Compute Wasserstein metrics on random grid
        metrics = compute_all_wasserstein_metrics(true_quantiles, predictions)

        # Compute additional metrics on fixed grid if available
        if fixed_predictions is not None and fixed_posterior_samples is not None:
            # Compute true quantiles for fixed grid
            true_quantiles_fixed = experiment_dgp.compute_true_quantiles(x_fixed, q_fixed, quantile_estimation_seed)

            # Compute coverage metrics for multiple alpha levels (default: 0.1, 0.25, 0.5)
            coverage_metrics = compute_coverage_metrics(true_quantiles_fixed, fixed_posterior_samples)

            # Compute pinball loss metrics (19 per-quantile + 1 CRPS)
            pinball_metrics = compute_pinball_metrics(true_quantiles_fixed, fixed_predictions)

            # Merge all metrics
            metrics.update(coverage_metrics)
            metrics.update(pinball_metrics)

        return {
            **metrics,
            'fit_predict_time': fit_predict_time,
            'exitcode': 0,
            'error': None
        }

    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        print(f"Full error traceback:\n{full_traceback}")
        return {
            'avg_wasserstein_1': np.nan,
            'avg_wasserstein_infty': np.nan,
            'sup_wasserstein_1': np.nan,
            'sup_wasserstein_infty': np.nan,
            'fit_predict_time': np.nan,
            'exitcode': 1,
            'error': str(e)
        }
