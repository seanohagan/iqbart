import pandas as pd
from typing import Dict, Any, List

def format_results_for_output(
    experiment_name: str,
    method_name: str,
    experiment_params: Dict[str, Any],
    method_params: Dict[str, Any],
    repetition: int,
    n_samples: int,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """Format experimental results for CSV output."""
    output_row = {
        'experiment': experiment_name,
        'method': method_name,
        'repetition': repetition,
        'n_samples': n_samples,
    }

    # Add method parameters with method prefix
    for param, value in method_params.items():
        output_row[f'method_{param}'] = value

    # Add experiment parameters with experiment prefix
    for param, value in experiment_params.items():
        if param not in ['domain', 'quantile_range']:  # Skip complex params
            output_row[f'experiment_{param}'] = value

    # Add results
    output_row.update(results)

    return output_row

def save_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save experimental results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def load_results_from_csv(input_path: str) -> pd.DataFrame:
    """Load experimental results from CSV file."""
    return pd.read_csv(input_path)