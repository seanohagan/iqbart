import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Try to import rpy2, with graceful fallback if not available
try:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    # Activate automatic conversion between R and NumPy
    rpy2.robjects.numpy2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("rpy2 not available. DRBART will not work. Please install with 'pip install rpy2'")

class DRBART:
    """
    Python interface to the R DRBART (Dirichlet Random Basis BART) package for quantile regression.

    This class provides an interface that's compatible with the IQF experiment
    framework while using the R DRBART package for fitting and prediction.
    """

    def __init__(self,
                 domain: np.ndarray,
                 n_trees: int = 200,
                 nburn: int = 100,
                 nsim: int = 100,
                 seed: Optional[int] = None,
                 ygrid_size: int = 100,
                 r_verbose: bool = False,
                 **kwargs):
        """
        Initialize the DRBART model.

        Args:
            domain: Array of shape (d, 2) representing the bounds of each feature
            n_trees: Number of trees (not directly used but kept for API compatibility)
            nburn: Number of burn-in MCMC iterations
            nsim: Number of post-burn MCMC iterations
            seed: Random seed for reproducibility
            ygrid_size: Number of points in y-grid for CDF estimation
            r_verbose: Whether to show verbose output from R
            **kwargs: Additional arguments for API compatibility
        """
        self.domain = domain
        self.nburn = nburn
        self.nsim = nsim
        self.n_trees = n_trees  # Not used directly but kept for compatibility
        self.seed = seed
        self.ygrid_size = ygrid_size
        self.r_verbose = r_verbose
        self.config = kwargs

        # Initialize R-related attributes
        self.r_model = None
        self.is_fitted = False

        # Check if R and drbart are available
        if not R_AVAILABLE:
            warnings.warn("R integration via rpy2 is not available. DRBART will not work.")
            return

        # Try to import the drbart package
        try:
            self.r_base = importr('base')
            self.r_drbart = importr('drbart')
            self.r_available = True
        except Exception as e:
            warnings.warn(f"Failed to import R drbart package: {e}. "
                         "Make sure it's installed in R with 'install.packages(\"drbart\")'")
            self.r_available = False

    def _determine_ygrid(self, y: np.ndarray) -> np.ndarray:
        """
        Determine a suitable y-grid for prediction based on training data.

        Args:
            y: Training response values

        Returns:
            Grid of y values covering the likely range of the response
        """
        y_min = np.min(y)
        y_max = np.max(y)

        # Add padding to ensure we cover the distribution tails
        padding = (y_max - y_min) * 0.3
        y_min -= padding
        y_max += padding

        return np.linspace(y_min, y_max, self.ygrid_size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the DRBART model to training data.

        Args:
            X: Array of shape (n_samples, n_features) containing training inputs
            y: Array of shape (n_samples,) containing target values

        Raises:
            RuntimeError: If R or drbart package are not available
        """
        if not hasattr(self, 'r_available') or not self.r_available:
            raise RuntimeError("R and/or drbart package not available. Cannot fit model.")

        # Store data for later use in prediction
        self.X_train = X
        self.y_train = y
        self.ygrid = self._determine_ygrid(y)

        # Convert data to R format (should be automatic with numpy2ri)
        r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        r_y = ro.FloatVector(y)

        # Set random seed if provided
        if self.seed is not None:
            ro.r('set.seed')(self.seed)

        # Call drbart function
        if self.r_verbose:
            print("Fitting DRBART model...")

        # Create the model call with appropriate parameters
        self.r_model = self.r_drbart.drbart(
            r_y, r_X,
            nburn=self.nburn,
            nsim=self.nsim
        )

        if self.r_verbose:
            print("DRBART model fitting complete.")

        self.is_fitted = True

    def estimate_quantile(self, ygrid: np.ndarray, cdf: np.ndarray, q: float) -> float:
        """
        Estimate quantile value using linear interpolation of CDF.

        Args:
            ygrid: Array of y-values
            cdf: Array of cumulative probabilities for each y-value
            q: Desired quantile (between 0 and 1)

        Returns:
            Estimated value of the q-th quantile
        """
        return np.interp(q, cdf, ygrid)

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
            RuntimeError: If R prediction fails
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

        # Handle output_shape parameter (for compatibility with experiment framework)
        if "output_shape" in kwargs:
            output_shape = kwargs["output_shape"]
            if isinstance(output_shape, int) and output_shape != n_quantiles:
                # Generate evenly spaced quantiles if needed
                if output_shape > n_quantiles:
                    quantiles = np.linspace(min(quantiles), max(quantiles), output_shape)
                    n_quantiles = output_shape
                    result = np.zeros((n_samples, n_quantiles))

        # Convert test data and ygrid to R format
        r_X_test = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        print(f"ygrid shape: {self.ygrid.shape}")
        r_ygrid = ro.FloatVector(self.ygrid)

        # Get distribution predictions from R
        try:
            if self.r_verbose:
                print("Getting predictions from DRBART...")

            # Call R predict function directly
            r_predict = ro.r('predict')
            raw_preds = r_predict(self.r_model, r_X_test, r_ygrid, type="distribution")

            # Extract the ygrid from element 4
            ygrid_used = np.array(raw_preds[4])

            # Extract predictions from element 0 (the 3D array)
            pred_array = raw_preds[0]

            # Convert to numpy array and process
            pred_np = np.array(pred_array)

            # Check if we have the expected 3D structure (n_samples, n_ygrid, n_mcmc_samples)
            if len(pred_np.shape) == 3 and pred_np.shape[0] >= n_samples:
                # Process each test point
                for i in range(n_samples):
                    # Get predictions for test point i
                    point_pred = pred_np[i]

                    # Average over MCMC samples (axis=1) to get CDF
                    cdf = np.mean(point_pred, axis=1)

                    # Calculate quantiles for each requested probability
                    for j, q in enumerate(quantiles):
                        result[i, j] = self.estimate_quantile(ygrid_used, cdf, q)
            else:
                # Fall back to a more general approach if structure is different
                if self.r_verbose:
                    print(f"Unexpected prediction structure. Shape: {pred_np.shape}")

                # Process each test point individually
                for i in range(min(n_samples, pred_np.shape[0])):
                    try:
                        # Try to extract and process predictions for this test point
                        pred_i = pred_np[i]

                        # Skip if it's not numeric
                        if isinstance(pred_i, str):
                            continue

                        # Convert to numpy and try to determine structure
                        if hasattr(pred_i, 'shape'):
                            if len(pred_i.shape) == 2:
                                # Average over first dimension
                                cdf = np.mean(pred_i, axis=0)
                            else:
                                # Use as is
                                cdf = pred_i

                            # Calculate quantiles
                            for j, q in enumerate(quantiles):
                                result[i, j] = self.estimate_quantile(ygrid_used, cdf, q)
                    except Exception as e:
                        if self.r_verbose:
                            print(f"Error processing test point {i}: {e}")

            # Handle NaN values
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                if self.r_verbose:
                    print(f"Replacing {np.sum(np.isnan(result) | np.isinf(result))} NaN/Inf values")

                # Replace NaNs with mean of non-NaN values
                valid_mask = ~(np.isnan(result) | np.isinf(result))
                if np.any(valid_mask):
                    valid_mean = np.mean(result[valid_mask])
                    result = np.where(np.isnan(result) | np.isinf(result), valid_mean, result)
                else:
                    # All values are NaN, replace with 0
                    result = np.zeros((n_samples, n_quantiles))

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
            raise RuntimeError(f"Error during DRBART prediction: {e}")


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                nburn: int = 200,
                nsim: int = 200,
                ygrid_size: int = 200,
                r_verbose: bool = False,
                **kwargs) -> np.ndarray:
    """Fit DRBART model and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    model = DRBART(
        domain=domain,
        nburn=nburn,
        nsim=nsim,
        ygrid_size=ygrid_size,
        r_verbose=r_verbose
    )

    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))

    return predictions