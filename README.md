# IQBART Experiments

Clean, functional experimental framework for distributional regression using Hydra.

## Installation

```bash
cd iqbart-experiments
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a single experiment:
```bash
python main.py experiment=cdmixture method=iqbart
```

Run with different parameters:
```bash
python main.py experiment=stealthy_chameleon method=gp repetition=5 n_samples=2000
```

### Available Experiments
- `cdmixture`: Covariate-dependent mixture model
- `stealthy_chameleon`: Complex multi-modal distribution
- `lstar`: Logistic Smooth Transition Autoregressive model

### Available Methods
- `iqbart`: Implicit Quantile BART (PyMC implementation)
- `iqbart_cpp`: Implicit Quantile BART (C++ implementation)
- `qrf`: Quantile Random Forest
- `gp`: Gaussian Process quantile regression
- `iqn`: Implicit Quantile Networks (JAX/Flax)
- `drbart`: Dirichlet Random BART (R-based)
- `bart`: Homoscedastic BART
- `hbart`: Heteroscedastic BART
- `gamlss`: GAMLSS (R-based)

### Configuration

Experiments and methods are configured via YAML files in `config/`:
- `config/experiment/`: Experiment-specific parameters
- `config/method/`: Method-specific hyperparameters

### Parameter Sweeps

Use Hydra's multirun for parameter sweeps:
```bash
python main.py --multirun experiment=cdmixture method=iqbart,gp repetition=0,1,2,3,4
```

### Cluster Integration

For SLURM clusters:
```bash
python main.py --multirun hydra/launcher=submitit_slurm \
  experiment=cdmixture method=iqbart repetition=0,1,2,3,4
```

## Output

Results are saved to CSV files with columns:
- Experimental metadata (experiment, method, repetition, n_samples)
- Method hyperparameters
- Four Wasserstein metrics
- Timing and exit code information

## Adding New Methods

1. Create `src/methods/new_method.py`
2. Implement `fit_predict(X_train, y_train, x_grid, q_grid, **hyperparams)` function
3. Add method config in `config/method/new_method.yaml`

## Adding New Experiments

1. Create `src/experiments/new_experiment.py`
2. Implement `generate_data(n_samples, seed)` and `compute_true_quantiles(x_grid, q_grid)` functions
3. Add experiment config in `config/experiment/new_experiment.yaml`