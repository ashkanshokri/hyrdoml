
# imports
from typing import Callable, Dict, Any
import numpy as np


def get_transformer(transform: str) -> Callable:
    
    if transform == 'norm':
        return NormTransformer
    
    else:
        raise ValueError(f"Unsupported transformer: {transform}")
    
class BaseTransformer(object):
    def __init__(self, ):
        self.parameters = {}

    def transform(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError


class NormTransformer(BaseTransformer):
    def __init__(self, parameters: Dict[str, Any]=None):
        self.parameters = parameters if parameters is not None else {}


    def calculate_parameters(self, x: np.ndarray):
        # ignore the nan values

        self.parameters['mean'] = np.nanmean(x).tolist()
        self.parameters['std'] = np.nanstd(x).tolist()
        return self.parameters

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("Parameters are not calculated")
        else:
            return (x - np.array(self.parameters['mean'])) / np.array(self.parameters['std'])

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("Parameters are not calculated")
        else:
            return x * np.array(self.parameters['std']) + np.array(self.parameters['mean'])

