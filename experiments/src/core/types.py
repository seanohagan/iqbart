from typing import Callable, Tuple, NamedTuple
import numpy as np

Method = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]

DataGenerator = Callable[[int, int], Tuple[np.ndarray, np.ndarray]]

QuantileComputer = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

class ExperimentDGP(NamedTuple):
    generate_data: DataGenerator
    compute_true_quantiles: QuantileComputer
