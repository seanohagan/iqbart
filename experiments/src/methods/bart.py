import numpy as np
import warnings
from typing import Dict, Any, Optional, List, Tuple

# Try to import pymc and pymc_bart
try:
    import pymc as pm
    import pymc_bart as pmb
    import pytensor.tensor as pt
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("pymc or pymc_bart not available. HomoscedasticBART will not work. Please install with 'pip install pymc pymc_bart'")

class HomoscedasticBART:
    """
    Bayesian Additive Regression Trees (BART) with homoscedastic Gaussian errors.

    This class provides an interface that's compatible with the IQF experiment
    framework while using regular BART regression with constant error variance.
    """

    def __init__(self,
                 domain: np.ndarray,
                 n_trees: int = 50,
                 n_chains: int = 2,
                 n_tune: int = 500,
                 n_draw: int = 500,
                 alpha: float = 0.95,
                 beta: float = 2.0,
                 seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize the HomoscedasticBART model.

        Args:
            domain: Array of shape (d, 2) representing the bounds of each feature
            n_trees: Number of trees in the BART model
            n_chains: Number of MCMC chains
            n_tune: Number of tuning samples (burn-in)
            n_draw: Number of posterior samples to draw
            alpha: Tree prior alpha parameter
            beta: Tree prior beta parameter
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for API compatibility
        """
        self.domain = domain
        self.n_trees = n_trees
        self.n_chains = n_chains
        self.n_tune = n_tune
        self.n_draw = n_draw
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.config = kwargs

        self.model = None
        self.idata = None
        self.is_fitted = False

        # Check if required packages are available
        if not PYMC_AVAILABLE:
            warnings.warn("PyMC and/or PyMC-BART are not available. HomoscedasticBART will not work.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the BART model to training data.

        Args:
            X: Array of shape (n_samples, n_features) containing training inputs
            y: Array of shape (n_samples,) containing target values

        Raises:
            RuntimeError: If PyMC or PyMC-BART packages are not available
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC and/or PyMC-BART packages not available. Cannot fit model.")

        # Store data for possible later use
        self.X_train = X
        self.y_train = y

        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Create and fit BART model
        with pm.Model() as self.model:
            # Create data containers for X and y
            X_data = pm.MutableData("X", X)

            # BART model for the mean
            mu = pmb.BART("mu", X_data, y, m=self.n_trees, alpha=self.alpha, beta=self.beta)

            # Homoscedastic error variance
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

            # Sample from the posterior
            self.idata = pm.sample(
                draws=self.n_draw,
                tune=self.n_tune,
                chains=self.n_chains,
                random_seed=self.seed
            )

        self.is_fitted = True

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict quantiles for test data.

        Args:
            X: Array of shape (n_samples, n_features) containing test inputs
            **kwargs: Additional arguments including:
                q: Quantile values to predict (required)
                output_shape: Optional shape parameter for output

        Returns:
            Array of shape (n_samples, len(q)) containing predicted quantiles

        Raises:
            ValueError: If model is not fitted or required arguments are missing
            RuntimeError: If prediction fails
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        # Get quantiles to predict
        if "q" not in kwargs:
            raise ValueError("Quantile values 'q' must be provided for prediction")

        quantiles = np.atleast_1d(kwargs["q"])

        # Initialize output array
        n_samples = X.shape[0]
        n_quantiles = len(quantiles)
        result = np.zeros((n_samples, n_quantiles))

        # Handle output_shape parameter
        if "output_shape" in kwargs:
            output_shape = kwargs["output_shape"]
            if isinstance(output_shape, int) and output_shape != n_quantiles:
                # Generate evenly spaced quantiles if needed
                if output_shape > n_quantiles:
                    quantiles = np.linspace(min(quantiles), max(quantiles), output_shape)
                    n_quantiles = output_shape
                    result = np.zeros((n_samples, n_quantiles))

        try:
            # Set new test data in the model
            with self.model:
                pm.set_data({"X": X})

                # Sample from the posterior predictive distribution
                posterior_pred = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["mu", "sigma"],
                    predictions=True
                )

            # Extract posterior mean
            mu_samples = posterior_pred.predictions["mu"].values  # Shape: (chain, draw, sample)

            # Get sigma from the original inference data (idata)
            sigma_samples = self.idata.posterior["sigma"].values  # Shape: (chain, draw)

            # Average across chains and draws for mean and sigma
            mu_mean = mu_samples.mean(axis=(0, 1))  # Shape: (sample,)
            sigma_mean = sigma_samples.mean()  # Scalar

            # Compute quantiles using normal distribution
            from scipy.stats import norm
            for i, q in enumerate(quantiles):
                result[:, i] = mu_mean + norm.ppf(q) * sigma_mean

            # Handle output_shape for downsampling case
            if "output_shape" in kwargs:
                output_shape = kwargs["output_shape"]
                if isinstance(output_shape, int) and output_shape < n_quantiles:
                    # Take evenly spaced indices
                    indices = np.linspace(0, n_quantiles-1, output_shape, dtype=int)
                    result = result[:, indices]

            return result

        except Exception as e:
            raise RuntimeError(f"Error during HomoscedasticBART prediction: {e}")


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                n_trees: int = 200,
                n_chains: int = 2,
                n_tune: int = 200,
                n_draw: int = 200,
                **kwargs) -> np.ndarray:
    """Fit homoscedastic BART model and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    model = HomoscedasticBART(
        domain=domain,
        n_trees=n_trees,
        n_chains=n_chains,
        n_tune=n_tune,
        n_draw=n_draw
    )

    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))

    return predictions
