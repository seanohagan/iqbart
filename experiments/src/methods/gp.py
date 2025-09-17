import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from scipy.stats import norm

def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                kernel_type: str = "matern",
                nu: float = 2.5,
                alpha: float = 1e-10,
                n_restarts_optimizer: int = 10,
                normalize_y: bool = True,
                random_state: int = None,
                **kwargs) -> np.ndarray:
    """Fit GP model and predict quantiles on evaluation grid."""

    if kernel_type.lower() == "matern":
        length_scale = [0.1] * X_train.shape[1]
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(nu=nu, length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
    else:
        length_scale = [0.1] * X_train.shape[1]
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=normalize_y,
        random_state=random_state,
        optimizer="fmin_l_bfgs_b"
    )

    gp.fit(X_train, y_train)

    mean, std = gp.predict(x_grid, return_std=True)

    n_samples = x_grid.shape[0]
    n_quantiles = len(q_grid)
    predictions = np.zeros((n_samples, n_quantiles))

    for i, q in enumerate(q_grid):
        predictions[:, i] = mean + norm.ppf(q) * std

    return predictions