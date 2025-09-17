import numpy as np
from typing import Dict

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

def compute_all_wasserstein_metrics(true_quantiles: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute all four Wasserstein metrics."""
    return {
        'avg_wasserstein_1': avg_wasserstein_1(true_quantiles, predictions),
        'avg_wasserstein_infty': avg_wasserstein_infty(true_quantiles, predictions),
        'sup_wasserstein_1': sup_wasserstein_1(true_quantiles, predictions),
        'sup_wasserstein_infty': sup_wasserstein_infty(true_quantiles, predictions)
    }