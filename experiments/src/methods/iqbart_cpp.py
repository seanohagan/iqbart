import numpy as np
import warnings

# Try to import the compiled C++ module with graceful fallback
try:
    import iqbart_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    warnings.warn("iqbart_cpp not available. IQBART_CPP will not work. Please compile the C++ module.")

class IQBARTCPP:
    """
    A Python wrapper for the C++ iqbart implementation that conforms to
    the experiment framework's fit/predict interface.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model by storing hyperparameters.

        This method accepts keyword arguments from the experiment runner and maps
        them to the parameter names expected by the C++ function.
        """
        # --- Hyperparameter Mapping ---
        # We map the Python-friendly names from your experiment to the
        # short variable names in the C++ function signature.
        self.m = kwargs.get('n_trees', 200)
        self.nd = kwargs.get('n_draw', 1000)
        self.burn = kwargs.get('n_tune', 500)
        self.tau = kwargs.get('tau', 0.95)
        self.nu = kwargs.get('nu', 3.0)
        self.lambda_val = kwargs.get('lambda_val', 0.9)
        self.alpha = kwargs.get('alpha', 0.25)
        self.mybeta = kwargs.get('beta', 0.8) # Note: 'mybeta' in C++

        self.phi = kwargs.get('phi', 1.0)

        # These parameters seem fixed or less commonly tuned, but we can set them.
        self.nm = kwargs.get('nm', 100)
        self.nkeeptrain = self.nd
        self.nkeeptest = self.nd
        self.nkeeptestme = self.nd
        self.nkeeptreedraws = self.nd
        self.printevery = kwargs.get('printevery', 50)
        self.data_aug = kwargs.get('data_aug', True)

        self.n_quantile_samples_per_datum = kwargs.get('n_quantile_samples_per_datum', 1)
        self.seed = kwargs.get('seed', None)
        self.n_chains = kwargs.get('n_chains', 4)

        # The fit method will store the training data here
        self.X_train_ = None
        self.y_train_ = None

        # Check if C++ module is available
        if not CPP_AVAILABLE:
            warnings.warn("C++ iqbart module is not available. IQBARTCPP will not work.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        'Fits' the model by storing the training data. The actual C++ call
        is deferred to the predict method, as it requires the test data.
        """
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ iqbart module not available. Cannot fit model.")

        self.X_train_ = np.repeat(X, self.n_quantile_samples_per_datum, axis=0)
        self.y_train_ = np.repeat(y, self.n_quantile_samples_per_datum, axis=0)

        print(f'nqs is {self.n_quantile_samples_per_datum}, len(X_tr) is {len(self.X_train_)}')
        # Return self to allow for method chaining, e.g., model.fit(X,y).predict(X_test)
        return self

    def predict(self, X_test: np.ndarray, q: np.ndarray, **kwargs) -> np.ndarray:
        """
        Makes predictions by calling the underlying C++ iqbart function.

        Args:
            X_test: The covariate data for which to make predictions.
            q: The quantile levels to predict.

        Returns:
            A numpy array of shape (n_samples, n_quantiles) with the predictions.
        """
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ iqbart module not available. Cannot make predictions.")

        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("The model must be fit before making predictions.")

        # Handle output_shape parameter (for compatibility with experiment framework)
        quantiles = np.asarray(q).reshape(-1)
        if "output_shape" in kwargs:
            output_shape = kwargs["output_shape"]
            if isinstance(output_shape, int) and output_shape != len(quantiles):
                # Generate evenly spaced quantiles if needed
                if output_shape > len(quantiles):
                    quantiles = np.linspace(min(quantiles), max(quantiles), output_shape)
                elif output_shape < len(quantiles):
                    # Take evenly spaced indices
                    indices = np.linspace(0, len(quantiles)-1, output_shape, dtype=int)
                    quantiles = quantiles[indices]

        # Ensure all inputs are float64, as expected by the C++ code
        x_train = np.ascontiguousarray(self.X_train_, dtype=np.float64)
        y_train = np.ascontiguousarray(self.y_train_, dtype=np.float64)
        x_test = np.ascontiguousarray(X_test, dtype=np.float64)
        q_test = np.ascontiguousarray(quantiles, dtype=np.float64)

        # Call the C++ function exposed by pybind11
        results = iqbart_cpp.iqbart(
            x=x_train,
            y=y_train,
            xp=x_test,
            qp=q_test,
            tau=self.tau,
            nu=self.nu,
            lambda_val=self.lambda_val,
            alpha=self.alpha,
            mybeta=self.mybeta,
            phi=self.phi,
            nd=int(self.nd),
            burn=int(self.burn),
            m=int(self.m),
            nm=int(self.nm),
            nkeeptrain=int(self.nkeeptrain),
            nkeeptest=int(self.nkeeptest),
            nkeeptestme=int(self.nkeeptestme),
            nkeeptreedraws=int(self.nkeeptreedraws),
            printevery=int(self.printevery),
            data_aug=self.data_aug,
            seed=self.seed if self.seed is not None else 42,
            num_chains=int(self.n_chains)
        )

        # The C++ module now returns a multi-chain result object with:
        # - results.num_chains: number of chains
        # - results (vector): each element is an IQBartResults for that chain
        n_samples = X_test.shape[0]
        n_quantiles = len(quantiles)

        # Collect predictions from all chains
        all_chain_predictions = []

        for chain_idx in range(results.num_chains):
            # Get results for this chain
            chain_results = results.chain_results[chain_idx]  # Access chain_idx-th IQBartResults

            # Convert yhat_test_mean to numpy array and reshape
            # Each chain has shape (n_samples * n_quantiles,)
            chain_predictions = np.array(chain_results.yhat_test_mean).reshape(n_samples, n_quantiles)
            all_chain_predictions.append(chain_predictions)

        # Stack predictions from all chains: shape (num_chains, n_samples, n_quantiles)
        all_chains = np.stack(all_chain_predictions, axis=0)

        # Average across chains to get final predictions: shape (n_samples, n_quantiles)
        predictions = np.mean(all_chains, axis=0)

        return predictions


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                n_trees: int = 500,
                alpha: float = 0.5,
                beta: float = 0.25,
                mybeta: float = 0.8,
                n_tune: int = 1000,
                n_draw: int = 1000,
                nm: int = 20,
                n_quantile_samples_per_datum: int = 1,
                phi: float = 1.0,
                tau: float = 0.95,
                nu: float = 3.0,
                lambda_val: float = 0.9,
                printevery: int = 50,
                data_aug: bool = True,
                n_chains: int = 4,
                seed: int = None,
                **kwargs) -> np.ndarray:
    """Fit IQBART C++ model and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    print(data_aug)

    model = IQBARTCPP(
        domain=domain,
        n_trees=n_trees,
        alpha=alpha,
        beta=beta,
        mybeta=mybeta,
        n_tune=n_tune,
        n_draw=n_draw,
        nm=nm,
        n_quantile_samples_per_datum=n_quantile_samples_per_datum,
        phi=phi,
        tau=tau,
        nu=nu,
        lambda_val=lambda_val,
        printevery=printevery,
        data_aug=data_aug,
        n_chains=n_chains,
        seed=seed
    )

    model.fit(X_train, y_train)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))
    print(np.sum(predictions))

    return predictions
