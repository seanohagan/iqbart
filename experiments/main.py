import hydra
from omegaconf import DictConfig
from src.core.harness import run_single_experiment
from src.evaluation.utils import format_results_for_output, save_results_to_csv
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for distributional regression experiments."""

    # try:
    #     import pytensor
    #     pytensor.config.cxx = cfg.get('cxx', '/usr/bin/clang++')
    # except ImportError:
    #     pass  # PyTensor not available, that's fine

    experiment_params = cfg.get('experiment_params', {})
    method_params = cfg.get('method_params', {})

    result = run_single_experiment(
        experiment_name=cfg.experiment,
        method_name=cfg.method,
        experiment_params=experiment_params,
        method_params=method_params,
        n_data=cfg.n_data,
        repetition=cfg.repetition,
        eval_grid_size=cfg.eval_grid_size,
        seed_offset=cfg.seed_offset
    )

    formatted_result = format_results_for_output(
        experiment_name=cfg.experiment,
        method_name=cfg.method,
        experiment_params=experiment_params,
        method_params=method_params,
        repetition=cfg.repetition,
        n_samples=cfg.n_data,
        results=result
    )

    output_file = os.path.join(os.getcwd(), "result.csv")
    save_results_to_csv([formatted_result], output_file)

    print(f"Experiment completed:")
    print(f"  Experiment: {cfg.experiment}")
    print(f"  Method: {cfg.method}")
    print(f"  Repetition: {cfg.repetition}")
    print(f"  N_data: {cfg.n_data}")
    print(f"  Results:")
    for metric in ['avg_wasserstein_1', 'avg_wasserstein_infty', 'sup_wasserstein_1', 'sup_wasserstein_infty']:
        print(f"    {metric}: {result.get(metric, 'N/A'):.6f}")
    print(f"  Fit+Predict time: {result.get('fit_predict_time', 'N/A'):.3f}s")
    print(f"  Exit code: {result.get('exitcode', 'N/A')}")
    if result.get('error'):
        print(f"  Error: {result.get('error')}")
    print(f"  Results saved to: {output_file}")

if __name__ == "__main__":
    main()
