# imports
from typing import List, Callable
import xarray as xr
import numpy as np


def dynamic_feature_mean(dataset: xr.Dataset, dynamic_feature: str = None) -> np.ndarray:
    if dynamic_feature is None:
        raise ValueError("Dynamic features are not specified")

    return np.nanmean(dataset['dynamic'].sel(dynamic_feature=dynamic_feature).values, axis=1)


def get_evolving_dynamic_feature_function(name: str) -> Callable:
    if name == 'dynamic_feature_mean':
        return dynamic_feature_mean        
    else:
        raise ValueError(f"Evolving dynamic feature function {name} not found")