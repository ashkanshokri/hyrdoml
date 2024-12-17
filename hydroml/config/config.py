# imports
from pathlib import Path
from hydroml.utils import helpers as h


# typing
from typing import Any, Dict, Optional
from hydroml.utils import config_path_handler as cp

# functions
def load_config(path: str) -> 'Config':
    """Loads the configuration from a YAML file into a Config object.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class with loaded configuration.
    """
    config = Config()
    config.load_from_yaml(path)
    return config


# class
class Config(object):
    def __init__(self, **kwargs: Optional[Dict[str, Any]]):
        """Initializes the Config object with default and provided configurations.

        Args:
            **kwargs: Additional configuration parameters.
        """
        
        # default config
        model_config = {
            'model': 'HydroLSTM',
            'lstm_dynamic_input_feature_latent_size': 1,
            'lstm_static_input_latent_size': 1,
            'lstm_hidden_size': 256,
            'lstm_target_size': 1,
            'lstm_layers': 1,
            'dropout_probability': 0.0,
            'initial_forget_bias': 0.5,
        }

        # Training Configuration
        training_config = {
            'lr': 1e-4,
            'weight_decay': 0.0001,
            'loss_fn': 'nse',
            'batch_size': 512,
            'max_epochs': 120,
            'check_val_every_n_epoch': 5,
            'gradient_clip_val': 0.0,
            'enable_progress_bar': True,

        }

        # Feature Configuration
        feature_config = {
            'dynamic_features': ['precipitation_AWAP', 'et_morton_wet_SILO'],
            'target_features': ['streamflow_mmd'],
            'static_features': [
                'catchment_area', 'mean_slope_pct', 'prop_forested', 'upsdist', 'strdensity',
                'strahler', 'frac_snow', 'p_seasonality', 'p_mean', 'pet_mean', 'aridity',
                'high_prec_freq', 'high_prec_dur'
            ],
            'evolving_static_features': {'dynamic_feature_mean': {'name': 'dynamic_feature_mean', 'dynamic_feature': ['precipitation_AWAP']}},
            'add_sin_cos_doy': True,
            'evolving_metadata': {'observed_target_std': {'target_features': ['streamflow_mmd']}},
            'seqlen': 365,
            'transform_predictor_dynamic': 'norm',
            'transform_target': 'norm',
        }

        # Dataset Configuration
        dataset_config = {'dataset_config': {
            'name': 'camels_aus_v1',
            'static_feature_path_key': 'camels_aus_attributes',
            'dir_key': 'camels_aus_v1',
            'awra_dir_key': 'awra_postprocessed',
        }}

        # Calibration, Validation, and Test Configuration
        cal_val_test_config = {
            'cal': {
                'periods': [['2011-01-01', '2017-01-01']],
                'catchment_ids': ['410730', '401009']
            },
            'val': {
                'periods': [['2017-01-01', '2022-05-30']],
                'catchment_ids': ['410730']
            },
            'test': {
                'periods': [['2011-01-01', '2011-12-01']],
                'catchment_ids': ['410730']
            },
            'drop_catchments_with_all_nans': True,
        }

        # Path Configuration
        path_config = {
            'parent_path_key': 'parent_path',
            'transform_parameter_path': 'params.yaml',
            'finetune_directory_name_base': 'finetune',
            'name': 'default',
            'version': 'none',
        }

        # Hardware and Device Configuration
        hardware_config = {
            'device': 'cpu',
            'dataloader_nworkers': 12,
            'dataloader_persistent_workers': True,
            '_platform': kwargs.pop('platform', 'win'),
        }

        # Finetune Configuration
        finetune_config = {
            'save_top_k': 1,
            'layers_to_finetune': None,
            'finetune_max_epochs': 15,
            'finetune_lr': 7e-5,
        }

        # Miscellaneous Configuration
        misc_config = {
            'number_of_time_output_timestep': 1,
            'is_train': True,
            'head_hidden_layers': [10],
        }

        default_config = {**model_config, **training_config, **feature_config, **dataset_config, **cal_val_test_config, **path_config, **hardware_config, **finetune_config, **misc_config}
        
        default_config.update(kwargs)
        # set platform to make sure path_mapping is updated
        self.platform = default_config['_platform']
        self.update(default_config)


    @property
    def platform(self) -> str:
        return self._platform

    @platform.setter
    def platform(self, value: str) -> None:
        self._platform = value
        self.path_mapping = cp.get_path_mapping_file_from_config(platform=value)

    @property
    def parent_path(self) -> str:
        return self.path_mapping[self.parent_path_key]

    @parent_path.setter
    def parent_path(self, value: str) -> None:
        self.path_mapping[self.parent_path_key] = value
    
    @property
    def finetune_directory(self) -> str:
        if self.layers_to_finetune is None:
            return f'{self.finetune_directory_name_base}_all'
        else:
            return f'{self.finetune_directory_name_base}_{"_".join(self.layers_to_finetune)}'
    
    def make_dirpath(self) -> None:
        (self.current_path / f'{self.version}').mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Path:
        
        
        return Path(self.parent_path) / self.name 

    def _get_path(self, key: str) -> str:
        if Path(self.get(key)).is_absolute():
            return Path(self.get(key))
        else:

            if self.version is None:
                raise ValueError(f"Version is not set. Please set the version before getting the path, using config.set_new_version_name().")

            return Path(self.current_path) / self.version / self.get(key)
        
    def get_transform_parameters_path(self) -> str:
        return self._get_path('transform_parameter_path')    
    

    def update(self, kwargs: Dict[str, Any]) -> None:
        """Updates the configuration with the provided key-value pairs.

        Args:
            kwargs (Dict[str, Any]): The configuration parameters to update.
        """
        self.__dict__.update({k: v if v != 'none' else None for k, v in kwargs.items()})  # make sure it translate 'none' -> None

    @property
    def lstm_static_input_size(self) -> int:
        return len(self.static_features) + len(self.evolving_static_features)
    
    @property
    def lstm_dynamic_input_feature_size(self) -> int:
        return len(self.dynamic_features) + (2 if self.add_sin_cos_doy else 0)


    def __setitem__(self, k: str, v: Any) -> None:
        """Sets a configuration parameter.

        Args:
            k (str): The key of the configuration parameter.
            v (Any): The value of the configuration parameter.
        """
        self.__dict__[k] = v

    def __repr__(self) -> str:
        """Returns a string representation of the configuration.

        Returns:
            str: The string representation of the configuration.
        """
        return self.__dict__.__repr__()
    
    def keys(self) -> Dict[str, Any]:
        """Returns the keys of the configuration.

        Returns:
            Dict[str, Any]: The keys of the configuration.
        """
        return self.__dict__.keys()


    def save_to_yaml(self) -> None:
        """Saves the current configuration to a YAML file.

        Args:
            filepath (str): The path where the YAML file will be saved.
        """
        
        self.name = str(self.name)
        
        # exclude private attributes
        to_be_saved = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        self.parent_path = str(self.parent_path)

        # save
        h.save_yaml(to_be_saved, self.current_path / f'{self.version}' / 'config.yaml')

    def load_from_yaml(self, filepath: str) -> None:
        """Loads configuration from a YAML file.

        Args:
            filepath (str): The path to the YAML configuration file.
        """
        config_data = h.read_yaml(filepath)
        self.__dict__.update(config_data)

    def get(self, *args: str) -> Optional[Any]:
        """Gets a configuration parameter by key.

        Args:
            *args (str): The key of the configuration parameter.

        Returns:
            Optional[Any]: The value of the configuration parameter, or None if not found.
        """
        return self.__dict__.get(*args)
    
    def set_new_version_name(self) -> None:
        self.version = h.get_version_name()
