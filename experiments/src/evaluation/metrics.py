import numpy as np
from typing import Dict, Union, List

def avg_wasserstein_1(true_quantiles: np.ndarray, predictions: np.ndarray) -> float:
    """Average Wasserstein-1 distance across all x points."""
    w1_per_x = np.mean(np.abs(true_quantiles - predictions), axis=1)
    return float(np.mean(w1_per_x))

def avg_wasserstein_infty(true_quantiles: np.ndarray, predictions: np.ndarray) -> float:
    """Average Wasserstein-infinity distance across all x points."""
    winf_per_x = np.max(np.abs(true_quantiles - predictions), axis=1)
    return float(np.mean(winf_per_x))

def sup_wasserstein_1(true_quantiles: np.ndarray, predictions: np.ndarray) -> float:
    """Supremum Wasserstein-1 distance across all x points."""
    w1_per_x = np.mean(np.abs(true_quantiles - predictions), axis=1)
    return float(np.max(w1_per_x))

def sup_wasserstein_infty(true_quantiles: np.ndarray, predictions: np.ndarray) -> float:
    """Supremum Wasserstein-infinity distance across all x points."""
    winf_per_x = np.max(np.abs(true_quantiles - predictions), axis=1)
    return float(np.max(winf_per_x))

def does_post_ci_cover(posterior_samples: np.ndarray, true_estimand: float, alpha: float) -> int:
    """Computes if the 1-alpha quantile-based credible interval covers the estimand  """
    return int((true_estimand >= np.quantile(posterior_samples, alpha/2)) & (true_estimand <= np.quantile(posterior_samples, 1-(alpha/2))))

def compute_coverage_metrics(true_quantiles: np.ndarray,
                           posterior_samples: np.ndarray,
                           alpha: Union[float, List[float]] = [0.1, 0.25, 0.5]) -> Dict[str, int]:
    """
    Compute coverage metrics for each (x_fixed, q_fixed) combination.

    Args:
        true_quantiles: (n_fixed_points, n_fixed_quantiles) - true quantile values
        posterior_samples: (n_chains, n_fixed_points, n_fixed_quantiles, n_draws) - posterior samples
        alpha: significance level(s) for credible intervals (default [0.1, 0.25, 0.5] for 90%, 75%, 50% CIs)

    Returns:
        Dict with keys like 'cover-FX0-q0.05-alpha0.1' and values 0/1 for coverage
    """
    # Convert alpha to list if it's a single float
    if isinstance(alpha, (int, float)):
        alpha_list = [alpha]
    else:
        alpha_list = alpha

    n_fixed_points, n_fixed_quantiles = true_quantiles.shape
    n_chains, _, _, n_draws = posterior_samples.shape

    # Flatten posterior samples across chains and draws: (n_fixed_points, n_fixed_quantiles, n_chains * n_draws)
    flat_samples = posterior_samples.reshape(n_chains * n_draws, n_fixed_points, n_fixed_quantiles).transpose(1, 2, 0)

    coverage_metrics = {}

    for i in range(n_fixed_points):
        for j in range(n_fixed_quantiles):
            # Get true quantile value and posterior samples for this (x, q) combination
            true_value = true_quantiles[i, j]
            samples = flat_samples[i, j, :]  # (n_chains * n_draws,) samples

            # Get actual quantile value - assume it's from fixed grid (0.05, 0.1, ..., 0.95)
            q_value = 0.05 + j * 0.05  # This assumes q_fixed = [0.05, 0.1, ..., 0.95]

            # Compute coverage for each alpha level
            for alpha_val in alpha_list:
                coverage = does_post_ci_cover(samples, true_value, alpha_val)
                metric_name = f"cover-FX{i}-q{q_value:.2f}-alpha{alpha_val}"
                coverage_metrics[metric_name] = coverage

    return coverage_metrics

def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    """Compute pinball loss (quantile loss) for a single prediction."""
    error = y_true - y_pred
    return max(quantile * error, (quantile - 1) * error)

def compute_pinball_metrics(true_quantiles: np.ndarray,
                          predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute pinball loss metrics on fixed grid.

    Args:
        true_quantiles: (n_fixed_points, n_fixed_quantiles) - true quantile values
        predictions: (n_fixed_points, n_fixed_quantiles) - predicted quantile values

    Returns:
        Dict with pinball loss per quantile and CRPS
    """
    n_fixed_points, n_fixed_quantiles = true_quantiles.shape

    # Fixed quantile values (0.05, 0.1, ..., 0.95)
    q_values = np.arange(0.05, 1.0, 0.05)

    pinball_metrics = {}
    all_losses = []

    # Compute pinball loss for each quantile
    for j in range(n_fixed_quantiles):
        q_val = q_values[j]
        losses_for_q = []

        # Compute pinball loss for this quantile at each X point
        for i in range(n_fixed_points):
            loss = pinball_loss(true_quantiles[i, j], predictions[i, j], q_val)
            losses_for_q.append(loss)
            all_losses.append(loss)  # Store for CRPS

        # Average over X points for this quantile
        mean_loss = np.mean(losses_for_q)
        pinball_metrics[f"pinball-q{q_val:.2f}"] = mean_loss

    # CRPS: average of all pinball losses across all (X, q) pairs
    crps = np.mean(all_losses)
    pinball_metrics["crps"] = crps

    return pinball_metrics

def compute_all_wasserstein_metrics(true_quantiles: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute all four Wasserstein metrics."""
    return {
        'avg_wasserstein_1': avg_wasserstein_1(true_quantiles, predictions),
        'avg_wasserstein_infty': avg_wasserstein_infty(true_quantiles, predictions),
        'sup_wasserstein_1': sup_wasserstein_1(true_quantiles, predictions),
        'sup_wasserstein_infty': sup_wasserstein_infty(true_quantiles, predictions)
    }
