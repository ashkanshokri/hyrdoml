import torch
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np
import xarray as xr
from hydroml.data.dataset import get_transformer
from hydroml.utils import helpers as h
from hydroml.config.config import Config


def process_dataloader(
    dataloader: Any, 
    model: Any
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Processes a dataloader to extract predictions, ground truth values, dates, 
    and catchment IDs using a given model.
    
    Args:
        dataloader (Any): The dataloader providing batches of data.
        model (Any): The model with a `predict_step` method.

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]: 
            A tuple containing:
            - Predictions as a NumPy array.
            - Ground truth values as a NumPy array.
            - Dates as a Pandas DatetimeIndex.
            - Catchment IDs as a list of strings.
    """
    # Initialize lists to collect batch-wise data
    predictions: List[torch.Tensor] = []
    y_values: List[torch.Tensor] = []
    dates: List[str] = []
    catchment_ids: List[str] = []

    for batch in dataloader:
        predictions.append(model.predict_step(batch, 0, 0).detach().cpu().numpy())
        y_values.append(batch['y'].detach().cpu().numpy())
        dates.append(batch['date'].detach().cpu().numpy())
        catchment_ids.append(batch['catchment_id'])


    predictions = np.concatenate(predictions, axis=0)
    y = np.concatenate(y_values, axis=0)
    dates = pd.to_datetime(np.concatenate(dates))
    catchment_ids = [y for x in catchment_ids for y in x]


    return predictions, y, dates, catchment_ids


def convert_processed_dataloader_to_xarray(
    predictions: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    catchment_ids: List[str]
) -> xr.Dataset:
    """
    Converts processed dataloader data to an xarray Dataset.
    """
    import xarray as xr
    ds = xr.Dataset(
        {
            'prediction': (['idx', 'lead_time', 'feature'], predictions),
            'y': (['idx', 'lead_time', 'feature'], y),
            'date': (['idx'], dates),
            'catchment_id': (['idx'], catchment_ids)

        }
    )
    ds = ds.set_index(idx=["date", "catchment_id"]).unstack("idx")

    ds.sortby(["date"])

    return ds

def inverse_transform_predictions(ds: xr.Dataset, config: Config) -> xr.Dataset:
    """Inverse transforms model predictions back to original scale using saved transformation parameters.
    
    Args:
        ds (xr.Dataset): Dataset containing model predictions
        config (Config): Configuration object containing transform parameters
        
    Returns:
        xr.Dataset: Dataset with predictions transformed back to original scale
    """
    parameters = h.read_yaml(config.get_transform_parameters_path())
    parameters = parameters[config.target_features[0]]
    transformer = get_transformer(config.transform_target)(parameters=parameters)
    ds = transformer.inverse_transform(ds)
    return ds


# merge functions and transform predictions
def process_and_convert_dataloader_to_xarray(
    dataloader: Any,
    model: Any,
    config: Config = None,
    transform: bool = True,
    clip_at_zero: bool = False
) -> xr.Dataset:
    
    # process dataloader - in transformed scale
    predictions, y, dates, catchment_ids = process_dataloader(dataloader, model)

    

    # convert to xarray dataset - in transformed scale
    ds = convert_processed_dataloader_to_xarray(predictions, y, dates, catchment_ids)
    
    # transform predictions back to original scale if requested
    if transform:
        if config is None:
            raise ValueError("Config must be provided when transform=True")
        ds = inverse_transform_predictions(ds, config)

    if clip_at_zero:
        ds['prediction'] = ds['prediction'].clip(0)

    return ds
