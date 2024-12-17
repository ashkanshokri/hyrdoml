# imports
## standard
from pathlib import Path

## external
import xarray as xr

## internal
import hydroml.training.finetune as ft
from hydroml.config.config import Config, load_config
from hydroml.models import get_model_from_path
from hydroml.models.evaluate_model import evaluate_model
from hydroml.models.get_model_from_path import get_model_from_path
from hydroml.training.train import train
from hydroml.utils import helpers as h


def train_evaluate(config: Config, save_results: bool = True):

    # train the model
    current_path , version_pre = train(config)

    model_path = Path(current_path) / version_pre
    _ = evaluate_model(model_path, check_point='last', save_results=save_results)

    #results_pretrained['simulation']
    #results_pretrained['metrics']

    return model_path

def train_finetune_evaluate(config: Config, save_results: bool = True):
    exp_base_path = train_evaluate(config, save_results=save_results)

    metrics_list = []
    for catchment_id in config.get('val')['catchment_ids']:
        model = get_model_from_path(exp_base_path, check_point='last')
        
        if config.layers_to_finetune is not None:
            model.freeze_all_layers()
            model.unfreeze_layers(config.layers_to_finetune)

        # Update config for the specific catchment
        config = model.config  # Alternatively, load config from file
        config = ft.update_config_for_per_catchment_finetune(
            config, 
            cal_catchment_ids=[catchment_id], 
            val_catchment_ids=[catchment_id], 
            max_epochs=config.finetune_max_epochs
        )

        # Perform fine-tuning        
        current_path_ft, version_pre_ft = ft.finetune(
            model, 
            config, 
            finetune_name=catchment_id
        )
        model_path_finetuned = Path(current_path_ft) / version_pre_ft
        
        # Evaluate the fine-tuned model
        results_finetuned = evaluate_model(model_path_finetuned, check_point='last')
        metrics_finetuned = results_finetuned['metrics']
        
        # Append metrics to the list with catchment ID as a dimension
        #metrics_finetuned = metrics_finetuned.assign_coords(catchment_id=catchment_id) # it might already have it!
        metrics_list.append(metrics_finetuned)
    
    # Concatenate metrics along the 'catchment_id' dimension
    if save_results and metrics_list:
        save_path = exp_base_path / 'results'
        save_path.mkdir(parents=True, exist_ok=True)
        
        all_metrics = xr.concat(metrics_list, dim='catchment_id')
        finetune_directory = getattr(config, 'finetune_directory', 'default_finetune_dir')
        all_metrics.to_netcdf(save_path / f'metrics_{finetune_directory}.nc')

    return exp_base_path

def finetune_evaluate(config: Config, save_results: bool = True):
    
    exp_base_path = config.current_path / config.version
    
    metrics_list = []
    for catchment_id in config.get('val')['catchment_ids']:
        model = get_model_from_path(exp_base_path, check_point='last')
        
        if config.layers_to_finetune is not None:
            model.freeze_all_layers()
            model.unfreeze_layers(config.layers_to_finetune)

        # Update config for the specific catchment
        config = model.config  # Alternatively, load config from file
        config = ft.update_config_for_per_catchment_finetune(
            config, 
            cal_catchment_ids=[catchment_id], 
            val_catchment_ids=[catchment_id], 
            max_epochs=config.finetune_max_epochs
        )

        # Perform fine-tuning
        
        current_path_ft, version_pre_ft = ft.finetune(
            model, 
            config, 
            finetune_name=catchment_id
        )
        model_path_finetuned = Path(current_path_ft) / version_pre_ft
        
        # Evaluate the fine-tuned model
        results_finetuned = evaluate_model(model_path_finetuned, check_point='last')
        metrics_finetuned = results_finetuned['metrics']
        
        # Append metrics to the list with catchment ID as a dimension
        #metrics_finetuned = metrics_finetuned.assign_coords(catchment_id=catchment_id) # it might already have it!
        metrics_list.append(metrics_finetuned)
    
    # Concatenate metrics along the 'catchment_id' dimension
    if save_results and metrics_list:
        save_path = exp_base_path / 'results'
        save_path.mkdir(parents=True, exist_ok=True)
        
        all_metrics = xr.concat(metrics_list, dim='catchment_id')
        finetune_directory = getattr(config, 'finetune_directory', 'default_finetune_dir')
        all_metrics.to_netcdf(save_path / f'metrics_{finetune_directory}.nc')

    return exp_base_path


if __name__ == '__main__':
    import argparse
    from typing import List
    
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('routine', type=str, choices=['train', 'finetune'], required=True, help='Which routine to run - train or finetune')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('kwargs', nargs=argparse.REMAINDER, help='Additional keyword arguments to update the config')
    parser.add_argument('--pretrained_model_path', type=str, required=False, help='Path to pretrained model')

    args = parser.parse_args()
    config = load_config(args.config)
    kwargs = h.parse_kwargs(args.kwargs)

    config.update(kwargs)

    if args.routine == 'train':
        train_evaluate(config)

    elif args.routine == 'train_finetune_evaluate':
        train_finetune_evaluate(config)




