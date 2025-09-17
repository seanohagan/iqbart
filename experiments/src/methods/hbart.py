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
    warnings.warn("pymc or pymc_bart not available. HeteroscedasticBART will not work. Please install with 'pip install pymc pymc_bart'")

class HeteroscedasticBART:
    """
    Bayesian Additive Regression Trees (BART) with heteroscedastic Gaussian errors.

    This class uses BART to model both the mean and the standard deviation
    of a Gaussian distribution as functions of the input features.
    It provides an interface compatible with the IQF experiment framework.
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
        Initialize the HeteroscedasticBART model.

        Args:
            domain: Array of shape (d, 2) representing the bounds of each feature
            n_trees: Number of trees in each BART model (for both mean and variance)
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
            warnings.warn("PyMC and/or PyMC-BART are not available. HeteroscedasticBART will not work.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the BART model to training data, modeling both mean and variance.

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

        # Number of observations
        n_obs = len(y)

        # Create and fit BART model for both mean and log-variance
        with pm.Model() as self.model:
            # Create data container for X
            X_data = pm.MutableData("X", X)

            # Single BART model with 2 outputs: mean and log-variance
            # First dimension (index 0) models the mean
            # Second dimension (index 1) models the log-variance (log(sigma²))
            w = pmb.BART("w", X=X_data, Y=y, m=self.n_trees, shape=(2, n_obs),
                         alpha=self.alpha, beta=self.beta)

            # Extract mean and standard deviation from BART outputs
            # w[0] is the mean
            mu = w[0]

            # w[1] models log-variance, so exp(w[1]) is variance, and sqrt(exp(w[1])) is std dev
            # Using exp to ensure positive std dev
            sigma = pm.math.sqrt(pm.math.exp(w[1]))

            # Likelihood: Normal distribution with heteroscedastic variance
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
            # Get predictions by sampling from the posterior predictive distribution
            with self.model:
                pm.set_data({"X": X})

                # Sample both BART outputs for mean and std
                posterior_pred = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=["w"],
                    predictions=True
                )

            # Extract posterior samples for both mean and log-variance
            w_samples = posterior_pred.predictions["w"].values  # Shape: (chain, draw, 2, n_samples)

            # Average across chains and draws
            # w_mean[0] gives mean predictions, w_mean[1] gives log-variance predictions
            w_mean = w_samples.mean(axis=(0, 1))  # Shape: (2, n_samples)

            # Extract the mean predictions
            mu_pred = w_mean[0]  # Shape: (n_samples,)

            # Convert log-variance to standard deviation
            # Apply softplus to ensure positivity
            sigma_pred = np.sqrt(np.exp(w_mean[1]))  # Shape: (n_samples,)

            # For each quantile, calculate the value using the normal distribution
            from scipy.stats import norm
            for i, q in enumerate(quantiles):
                # Using the predicted mean and std for each test point
                result[:, i] = mu_pred + norm.ppf(q) * sigma_pred

            # Handle output_shape for downsampling case
            if "output_shape" in kwargs:
                output_shape = kwargs["output_shape"]
                if isinstance(output_shape, int) and output_shape < n_quantiles:
                    # Take evenly spaced indices
                    indices = np.linspace(0, n_quantiles-1, output_shape, dtype=int)
                    result = result[:, indices]

            return result

        except Exception as e:
            raise RuntimeError(f"Error during HeteroscedasticBART prediction: {e}")


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                n_trees: int = 200,
                n_chains: int = 2,
                n_tune: int = 200,
                n_draw: int = 200,
                **kwargs) -> np.ndarray:
    """Fit heteroscedastic BART model and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    model = HeteroscedasticBART(
        domain=domain,
        n_trees=n_trees,
        n_chains=n_chains,
        n_tune=n_tune,
        n_draw=n_draw
    )

    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))

    return predictions