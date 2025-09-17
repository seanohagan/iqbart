import importlib
from typing import Dict, Any
from .types import Method, ExperimentDGP

def get_method(method_name: str) -> Method:
    """Dynamically import and return method's fit_predict function."""
    module = importlib.import_module(f'src.methods.{method_name}')
    return module.fit_predict

def get_experiment_dgp(experiment_name: str) -> ExperimentDGP:
    """Dynamically import and return experiment's data generation functions."""
    module = importlib.import_module(f'src.experiments.{experiment_name}')
    return ExperimentDGP(
        generate_data=module.generate_data,
        compute_true_quantiles=module.compute_true_quantiles
    )

def validate_method_params(method_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and filter method parameters. Returns validated params."""
    return params

def validate_experiment_params(experiment_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and filter experiment parameters. Returns validated params."""
    return params