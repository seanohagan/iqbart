import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pymc as pm
import pymc_bart as pmb
import pytensor.tensor as pt
import pytensor
import arviz as az


def _create_posterior_mean_idata(original_idata):
    """Creates idata object when using "mean" method for sampling strategy."""
    posterior_dict = {}
    for var in original_idata.posterior.data_vars:
        posterior_dict[var] = original_idata.posterior[var].mean(dim=["chain", "draw"]).values
        posterior_dict[var] = np.expand_dims(posterior_dict[var], axis=(0, 1))

    coords = {
        "chain": [0],
        "draw": [0],
    }
    for coord in original_idata.posterior.coords:
        if coord not in ["chain", "draw"]:
            coords[coord] = original_idata.posterior.coords[coord]

    new_idata = az.InferenceData(
        posterior=az.dict_to_dataset(
            posterior_dict,
            coords=coords,
            dims={var: ["chain", "draw"] + list(original_idata.posterior[var].dims[2:]) for var in posterior_dict}
        )
    )

    return new_idata

def _prepare_test_data(X_test, quantiles):
    """Prepare test data for BART quantile prediction."""
    X_test = np.asarray(X_test)
    q = np.asarray(quantiles).reshape(-1)

    X_repeated = np.repeat(X_test, len(quantiles), axis=0)
    Q_test = np.tile(q, len(X_test))

    return np.column_stack((X_repeated, Q_test))

def _reshape_predictions(predictions, n_X_test, n_quantiles):
    """Reshape predictions back to (n_X_test, n_quantiles) shape."""
    return np.asarray(predictions).reshape(n_X_test, n_quantiles)

def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                n_trees: int = 200,
                n_quantile_samples_per_datum: int = 1,
                alpha: float = 0.95,
                beta: float = 1.0,
                n_tune: int = 100,
                n_draw: int = 100,
                hn_scale: float = 1.0,
                posterior_predictive_method: str = "mean",
                sig_random: bool = True,
                learning_rate: float = 1,
                **kwargs) -> np.ndarray:
    """Fit IQBART model and predict on evaluation grid."""

    if posterior_predictive_method not in ["mean", "sample"]:
        raise ValueError('posterior_predictive_method must be set to "mean" or "sample"')

    X_train_rep = np.repeat(X_train, n_quantile_samples_per_datum, axis=0)
    y_train_rep = np.repeat(y_train, n_quantile_samples_per_datum, axis=0)

    qarr = np.random.uniform(size=X_train_rep.shape[0]).reshape(-1,1)
    X_aug_train = np.hstack((X_train_rep, qarr))

    with pm.Model() as model:
        X_aug = pm.MutableData("X", X_aug_train)
        y = y_train_rep
        mu = pmb.BART("mu", X_aug, y, m=n_trees, alpha=alpha, beta=beta, separate_trees=True)

        if sig_random:
            sig = pm.HalfNormal("sig", hn_scale)
            obs = pm.AsymmetricLaplace("obs", mu=mu, b=sig, q=X_aug[:,-1], observed=y, shape=X_aug.shape[0])
        else:
            obs = pm.AsymmetricLaplace("obs", mu=mu, b=learning_rate, q=X_aug[:,-1], observed=y, shape=X_aug.shape[0])

        idata = pm.sample(draws=n_draw, tune=n_tune)

    X_test_aug = _prepare_test_data(x_grid, q_grid)

    if posterior_predictive_method == "mean":
        pmean_idata = _create_posterior_mean_idata(idata)
    elif posterior_predictive_method == "sample":
        pmean_idata = idata
    else:
        raise ValueError("posterior_predictive_method is invalid.")

    with model:
        pm.set_data({"X": X_test_aug})
        ppc = pm.sample_posterior_predictive(trace=pmean_idata,
                                             var_names=["mu"], predictions=True)

    preds = ppc.predictions["mu"].mean(("chain", "draw")).values
    res_preds = _reshape_predictions(preds, len(x_grid), len(q_grid))

    return res_preds