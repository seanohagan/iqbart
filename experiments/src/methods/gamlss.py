import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Try to import rpy2, with graceful fallback if not available
try:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector
    # Activate automatic conversion between R and NumPy
    rpy2.robjects.numpy2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("rpy2 not available. GAMLSS will not work. Please install with 'pip install rpy2'")

class GAMLSS:
    """
    Python interface to the R GAMLSS (Generalized Additive Models for Location, Scale and Shape) package.

    This class provides an interface compatible with the IQF experiment
    framework while using the R GAMLSS package for fitting and prediction.
    """

    def __init__(self,
                 domain: np.ndarray,
                 distribution: str = "NO",  # Normal distribution
                 mu_formula: str = None,
                 sigma_formula: str = None,
                 max_iterations: int = 100,
                 r_verbose: bool = False,
                 **kwargs):
        """
        Initialize the GAMLSS model.

        Args:
            domain: Array of shape (d, 2) representing the bounds of each feature
            distribution: Distribution to use, defaults to "NO" (Normal)
            mu_formula: Formula for the mean model, default is "~pb(x1)" for 1D or "~pb(x1)+pb(x2)+..." for multi-D
            sigma_formula: Formula for the sigma model, default is same as mu_formula
            max_iterations: Maximum number of iterations for fitting
            r_verbose: Whether to show verbose output from R
            **kwargs: Additional arguments for API compatibility
        """
        self.domain = domain
        self.distribution = distribution
        self.mu_formula = mu_formula
        self.sigma_formula = sigma_formula
        self.max_iterations = max_iterations
        self.r_verbose = r_verbose
        self.config = kwargs

        # Initialize R-related attributes
        self.r_model = None
        self.is_fitted = False

        # Check if R and gamlss are available
        if not R_AVAILABLE:
            warnings.warn("R integration via rpy2 is not available. GAMLSS will not work.")
            return

        # Try to import the gamlss packages
        try:
            self.r_base = importr('base')
            self.r_stats = importr('stats')
            self.r_gamlss = importr('gamlss')
            self.r_gamlss_dist = importr('gamlss.dist')
            self.r_available = True

            # Set default formulas if not provided
            if self.mu_formula is None or self.sigma_formula is None:
                # Will create actual formulas during fit when we know the dimensions
                pass

        except Exception as e:
            warnings.warn(f"Failed to import R gamlss packages: {e}. "
                         "Make sure they're installed in R with 'install.packages(c(\"gamlss\", \"gamlss.dist\"))'")
            self.r_available = False

    def _create_data_frame(self, X: np.ndarray, y: np.ndarray = None) -> ro.DataFrame:
        """
        Create an R data frame from X and optionally y.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Optional target values of shape (n_samples,)

        Returns:
            An R DataFrame object
        """
        # Create column names for X
        feature_names = [f"x{i+1}" for i in range(X.shape[1])]

        # Create a dictionary for the data frame
        data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}

        # Add y if provided
        if y is not None:
            data_dict["y"] = y

        # Convert to R data frame
        data_frame = ro.DataFrame(data_dict)

        return data_frame

    def _create_formulas(self, X: np.ndarray) -> Tuple[str, str]:
        """
        Create default formulas for mu and sigma if not provided.

        Args:
            X: Input features to determine dimensionality

        Returns:
            Tuple of (mu_formula, sigma_formula)
        """
        if self.mu_formula is not None and self.sigma_formula is not None:
            return self.mu_formula, self.sigma_formula

        # Create default formulas with penalized B-splines (pb) for each feature
        n_features = X.shape[1]
        feature_terms = [f"pb(x{i+1})" for i in range(n_features)]
        formula = "~" + "+".join(feature_terms)

        # Use the same formula for both mu and sigma by default
        return formula, formula

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GAMLSS model to training data.

        Args:
            X: Array of shape (n_samples, n_features) containing training inputs
            y: Array of shape (n_samples,) containing target values

        Raises:
            RuntimeError: If R or gamlss package are not available
        """
        if not hasattr(self, 'r_available') or not self.r_available:
            raise RuntimeError("R and/or gamlss package not available. Cannot fit model.")

        # Store data for potential later use
        self.X_train = X
        self.y_train = y

        # Create data frame
        r_data = self._create_data_frame(X, y)

        # Create formulas if needed
        mu_formula, sigma_formula = self._create_formulas(X)

        if self.r_verbose:
            print(f"Using formulas: mu={mu_formula}, sigma={sigma_formula}")

        # Convert formulas to R formulas
        r_mu_formula = ro.Formula(mu_formula)
        r_sigma_formula = ro.Formula(sigma_formula)

        # Create the model call
        if self.r_verbose:
            print(f"Fitting GAMLSS model with distribution={self.distribution}, "
                  f"mu_formula={mu_formula}, sigma_formula={sigma_formula}")

        try:
            # Debug: print the R data frame structure
            if self.r_verbose:
                print("R data frame structure:")
                ro.r("function(df) { print(head(df)); print(str(df)) }")(r_data)

            # Create formula strings based on dimensions
            if X.shape[1] == 1:
                main_formula = "y ~ pb(x1)"
                sigma_formula_str = "~pb(x1)"
            elif X.shape[1] == 2:
                main_formula = "y ~ pb(x1) + pb(x2)"
                sigma_formula_str = "~pb(x1) + pb(x2)"
            else:
                # For more than 2 dimensions, build formulas programmatically
                feature_terms = [f"pb(x{i+1})" for i in range(X.shape[1])]
                main_formula = "y ~ " + " + ".join(feature_terms)
                sigma_formula_str = "~ " + " + ".join(feature_terms)

            if self.r_verbose:
                print(f"Using formula: {main_formula}")
                print(f"Using sigma_formula: {sigma_formula_str}")

            # Call gamlss function in R, exactly like in the R code
            self.r_model = self.r_gamlss.gamlss(
                formula=ro.Formula(main_formula),
                sigma_formula=ro.Formula(sigma_formula_str),
                family=ro.r(f"{self.distribution}()"),  # Note: call NO() directly like in R code
                data=r_data,
                trace=self.r_verbose,
                c_iter=self.max_iterations
            )

            if self.r_verbose:
                print("GAMLSS model fitting complete.")
                print(self.r_base.summary(self.r_model))

            self.is_fitted = True

        except Exception as e:
            raise RuntimeError(f"Error fitting GAMLSS model: {e}")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict quantiles for test data.

        Args:
            X: Array of shape (n_samples, n_features) containing test inputs
            **kwargs: Additional arguments including:
                q: Quantile values to predict (required)
                output_shape: Optional shape parameter for output
                y_true: Optional true values for computing CRPS

        Returns:
            Array of shape (n_samples, len(q)) containing predicted quantiles

        Raises:
            ValueError: If model is not fitted or required arguments are missing
            RuntimeError: If R prediction fails
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        # Get quantiles to predict
        if "q" not in kwargs:
            raise ValueError("Quantile values 'q' must be provided for prediction")

        quantiles = np.asarray(kwargs["q"]).reshape(-1)  # Convert to 1D array

        # Initialize output array
        n_samples = X.shape[0]
        n_quantiles = len(quantiles)
        result = np.zeros((n_samples, n_quantiles))

        # Handle output_shape parameter (for compatibility with experiment framework)
        if "output_shape" in kwargs:
            output_shape = kwargs["output_shape"]
            if isinstance(output_shape, int) and output_shape != n_quantiles:
                # Generate evenly spaced quantiles if needed
                if output_shape > n_quantiles:
                    quantiles = np.linspace(min(quantiles), max(quantiles), output_shape)
                    n_quantiles = output_shape
                    result = np.zeros((n_samples, n_quantiles))

        try:
            # Create R data frame for prediction
            r_pred_data = self._create_data_frame(X)

            if self.r_verbose:
                print(f"Predicting quantiles for {n_samples} points.")

            # Get predictions for mu and sigma
            r_predict = ro.r('predict')
            pred_mu = r_predict(self.r_model, newdata=r_pred_data, type="response", what="mu")
            pred_sigma = r_predict(self.r_model, newdata=r_pred_data, type="response", what="sigma")

            if self.r_verbose:
                # Print some prediction values for debugging
                mu_values = np.array(pred_mu)
                sigma_values = np.array(pred_sigma)
                print(f"Mu range: [{mu_values.min():.4f}, {mu_values.max():.4f}]")
                print(f"Sigma range: [{sigma_values.min():.4f}, {sigma_values.max():.4f}]")

            # Convert to numpy arrays
            mu = np.array(pred_mu)
            sigma = np.array(pred_sigma)

            # If true values are provided, compute CRPS
            if "y_true" in kwargs and self.distribution == "NO":
                from scipy.stats import norm
                y_true = kwargs["y_true"]

                # Compute CRPS for Gaussian distribution
                def crps_norm(y, mu, sigma):
                    z = (y - mu) / sigma
                    return sigma * (z * (2 * norm.cdf(z) - 1) +
                                    2 * norm.pdf(z) - 1/np.sqrt(np.pi))

                crps_values = np.array([crps_norm(y_true[i], mu[i], sigma[i])
                                        for i in range(len(y_true))])
                mean_crps = np.mean(crps_values)

                if self.r_verbose:
                    print(f"Mean CRPS: {mean_crps:.4f}")

                # Store CRPS for reference
                self.last_crps = mean_crps

            # Calculate quantiles for each requested probability
            if self.distribution == "NO":  # Normal distribution
                from scipy.stats import norm
                for i, q in enumerate(quantiles):
                    result[:, i] = mu + norm.ppf(q) * sigma
            else:
                # Get the appropriate quantile function for the distribution from R
                r_q_func = ro.r(f"gamlss.dist::q{self.distribution}")

                for i, q in enumerate(quantiles):
                    # Call R's quantile function with our parameters
                    quant_values = r_q_func(q, mu=mu, sigma=sigma)
                    result[:, i] = np.array(quant_values)

            # Handle output_shape for downsampling case
            if "output_shape" in kwargs:
                output_shape = kwargs["output_shape"]
                if isinstance(output_shape, int) and output_shape < n_quantiles:
                    # Take evenly spaced indices
                    indices = np.linspace(0, n_quantiles-1, output_shape, dtype=int)
                    result = result[:, indices]

            return result

        except Exception as e:
            if self.r_verbose:
                print(f"Error in prediction: {e}")
            raise RuntimeError(f"Error during GAMLSS prediction: {e}")

    def predict_distribution(self, X: np.ndarray, grid_size: int = 200):
        """
        Generate full conditional distributions for visualization.

        Args:
            X: Array of shape (n_samples, n_features) containing test inputs
            grid_size: Number of points in the grid for each distribution

        Returns:
            Dictionary containing:
                'x_grid': Grid of values for the response variable
                'densities': Array of density values for each point in X
                'mu': Predicted mean for each point in X
                'sigma': Predicted standard deviation for each point in X
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

        try:
            # Create R data frame for prediction
            r_pred_data = self._create_data_frame(X)

            # Get predictions for mu and sigma
            r_predict = ro.r('predict')
            pred_mu = r_predict(self.r_model, newdata=r_pred_data, type="response", what="mu")
            pred_sigma = r_predict(self.r_model, newdata=r_pred_data, type="response", what="sigma")

            # Convert to numpy arrays
            mu = np.array(pred_mu)
            sigma = np.array(pred_sigma)

            # Create grid for distributions
            y_min = np.min(self.y_train) - 2 * np.std(self.y_train)
            y_max = np.max(self.y_train) + 2 * np.std(self.y_train)
            y_grid = np.linspace(y_min, y_max, grid_size)

            # Calculate densities for each point
            from scipy.stats import norm
            densities = np.zeros((len(X), grid_size))

            for i in range(len(X)):
                densities[i] = norm.pdf(y_grid, loc=mu[i], scale=sigma[i])

            return {
                'y_grid': y_grid,
                'densities': densities,
                'mu': mu,
                'sigma': sigma
            }

        except Exception as e:
            if self.r_verbose:
                print(f"Error in distribution prediction: {e}")
            raise RuntimeError(f"Error during GAMLSS distribution prediction: {e}")


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                distribution: str = "NO",
                max_iterations: int = 100,
                r_verbose: bool = False,
                **kwargs) -> np.ndarray:
    """Fit GAMLSS model and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    model = GAMLSS(
        domain=domain,
        distribution=distribution,
        max_iterations=max_iterations,
        r_verbose=r_verbose
    )

    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))

    return predictions