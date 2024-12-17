from typing import Any

from hydroml.config.config import Config
from hydroml.data import get_dataset
from hydroml.data.catchment import Catchment
from hydroml.data.dataset import Dataset
from hydroml.models.get_model_from_path import get_model_from_path
from hydroml.training import finetune as ft
from hydroml.training.trainer import get_trainer
from hydroml.utils import helpers as h



def update_dirpath_config_for_finetune(config: Config, finetune_name: str = 'default'):
    # update config
    # make sure transform_parameter_path absolute
    config.transform_parameter_path = str(config.get_transform_parameters_path())

    # update config
    config.name = f'{config.name}//{config.version}//{config.finetune_directory}//{finetune_name}'
    config.version = h.get_version_name()    
    config.make_dirpath()
    return config


def finetune(model: Any, config: Config, cal_split_name: str = 'cal', val_split_name: str = 'val', finetune_name: str = 'default'):
    '''
    fine-tune a pre-trained model on a new split.
    The method is not loading all configuration. and not build the model again. Should be used with caution.        
    '''

    # update config
    config = update_dirpath_config_for_finetune(config,  finetune_name) # might be need to  move to a better place

    # load data
    dataset_cal = get_dataset(config, cal_split_name, is_train=True)
    dataset_val = get_dataset(config, val_split_name, is_train=True)
    cal_dataloader = dataset_cal.to_dataloader()
    val_dataloader = dataset_val.to_dataloader()
    
    # save config
    config.save_to_yaml()

    # fine-tune
    trainer = get_trainer(config)

    model.lr = config.finetune_lr
    trainer.fit(model, cal_dataloader, val_dataloader)

    return config.current_path, config.version


def update_config_for_per_catchment_finetune(config: Config, cal_catchment_ids: list, val_catchment_ids: list, **kwargs) -> Config:
    '''
    update config for per catchment fine-tuning
    # TODO: change cal and val splits
    '''

    # Update calibration catchments
    config.cal = {**config.cal, 'catchment_ids': cal_catchment_ids}

    # Update validation catchments 
    config.val = {**config.val, 'catchment_ids': val_catchment_ids}

    
    # update other configs
    config.update(kwargs)

    return config


def update_config_for_weighted_multiple_catchment_finetune(config: Config) -> Config:
    '''
    update config for multiple catchment fine-tuning
    '''

    return config



def run_finetune_from_timeseries(model_path, dynamic_data, static_data, catchment_id, **kwargs):
    """
    Runs a hydrological simulation using the given model path, dynamic data, and static data.

    Parameters:
    - model_path (str): Path to the model.
    - dynamic_data (pd.DataFrame): Time-indexed dynamic forcing data (e.g., precipitation, temperature and streamflow). The dynamic data should contain the target data.
    - static_data (list): Static features for the catchment (e.g., catchment area, mean slope).
    - catchment_id (str): Identifier for the catchment.
    

    Returns:
    - pd.Series: Simulation results as a time series of predictions.
    """
    # Load the model
    model = get_model_from_path(model_path)
    
    model.config.parent_path = str(model_path.parent.parent)

    # make sure the name and the version are correct - this is helpful in case the model file name is changed.
    model.config.name = model_path.parent.name
    model.config.version = model_path.name
    model.config.transform_parameter_path = 'params.yaml'

    config = model.config

    # Set up configuration for testing
    config.test = {
        'catchment_ids': [catchment_id],
        'periods': [[dynamic_data.index[0].strftime('%Y-%m-%d'), dynamic_data.index[-1].strftime('%Y-%m-%d')]]
    }

    config.update(kwargs)

    if isinstance(static_data, dict):
        static_data = [static_data[x] for x in config.static_features]
    # Create Catchment object
    catchments = [Catchment(dynamic_data, static_data, id=catchment_id, metadata={})]

    print(config.transform_parameter_path)
    # Create Dataset object
    dataset = Dataset.from_catchments(catchments, config, 'test', is_train=False)

    # Convert Dataset to DataLoader
    dataloader = dataset.to_dataloader()
    

    config = ft.update_dirpath_config_for_finetune(config, finetune_name=catchment_id)
    model.lr = config.finetune_lr
    if config.layers_to_finetune is not None:
        model.freeze_all_layers()
        model.unfreeze_layers(config.layers_to_finetune)
    
    # fine-tune
    config.max_epochs = config.finetune_max_epochs
    trainer = get_trainer(config)

    
    config.save_to_yaml()
    
    trainer.fit(model, dataloader)



    return config.current_path, config.version
