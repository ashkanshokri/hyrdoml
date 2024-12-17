from pathlib import Path
from hydroml.models.get_model_from_path import get_model_from_path
from hydroml.data import get_dataset
from hydroml.prediction.prediction import process_and_convert_dataloader_to_xarray
from hydroml.evaluation.metrics import get_metrics
import xarray as xr
import pytorch_lightning as pl
from typing import Union, Dict

def evaluate_model(model: Union[Path, pl.LightningModule], check_point: str = 'last', save_results: bool = True) -> Dict[str, xr.Dataset]:
    if isinstance(model, Path) or isinstance(model, str):
        model = get_model_from_path(model, check_point=check_point)
    elif isinstance(model, pl.LightningModule):
        model = model
    else:
        raise ValueError(f'model must be a Path or a LightningModule, got {type(model)}')
    
    dataset = get_dataset(model.config, 'val', is_train=False)
    dataloader = dataset.to_dataloader()

    ds = process_and_convert_dataloader_to_xarray(dataloader, model, model.config)

    if save_results:
        path = model.config.current_path / model.config.version
        save_path = path/'results'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        ds.to_netcdf(save_path/'simulation.nc')
        metrics = get_metrics(ds)
        metrics.to_netcdf(save_path/'metrics.nc')

    return {'simulation':ds, 'metrics': metrics}