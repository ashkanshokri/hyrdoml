# Configuration

The **Config** module provides a centralized mechanism to define, load, and manage configurations for training and validation of HydroLSTM models. This module simplifies model configuration by organizing various aspects—such as model architecture, training parameters, feature specifications, dataset details, and hardware configurations—into a single, manageable object.

## Core Functions and Classes

### `load_config`

Loads configuration parameters from a YAML file into a `Config` object.

```python
def load_config(path: str) -> 'Config':
    """Loads the configuration from a YAML file into a Config object.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class with loaded configuration.
    """
```

---

### `Config` Class

The `Config` class defines default configurations for all aspects of the HydroLSTM model.

#### **Initialization**

The constructor initializes default configurations, which can be overridden via keyword arguments. Simply add any parameter to the constructor to override the default configuration.

```python
class Config(object):
    def __init__(self, **kwargs: Optional[Dict[str, Any]]):
        """Initializes the Config object with default and provided configurations."""
```

#### Default Configurations

To make it easier to read, the configuration is divided into the following categories:

1. **Model Configuration:**
   - Defines LSTM architecture parameters (e.g., hidden size, number of layers, dropout probability).

2. **Training Configuration:**
   - Specifies training hyperparameters (e.g., learning rate, batch size, loss function).

3. **Feature Configuration:**
   - Defines static and dynamic features used for model inputs.
   - Allows transformations, such as normalization and adding sinusoidal day-of-year features.

4. **Dataset Configuration:**
   - Contains details about dataset structure and paths.

5. **Calibration, Validation, and Testing:**
   - Defines periods and catchment IDs for calibration, validation, and testing.

6. **Path Configuration:**
   - Manages paths for saving parameters, outputs, and model versions.

7. **Hardware Configuration:**
   - Controls device setup (e.g., CPU/GPU) and data loader optimizations.

8. **Fine-Tuning Configuration:**
   - Configures parameters for fine-tuning pre-trained models.

9. **Miscellaneous:**
   - Includes additional settings (e.g., output timesteps, training mode).

#### **Dynamic Properties**

The class provides dynamic attributes for derived configurations:

- `platform`: Manages path mappings for different computing environments (e.g., Windows, Linux). Key features:
  - Automatically updates path mappings when platform changes
  - Uses JSON configuration files in `config/path_mapping/` to define paths per platform
  - Example usage:
    ```python
    # Load config for Windows environment
    config = Config(platform='win') 
    
    # Switch to virga environment paths
    config.platform = 'virga'
    ```
  - Useful for maintaining consistent paths across development and production environments
  - Handled internally by `hydroml.config_path_handler`
- `parent_path`: Retrieves the root directory path.
- `current_path`: Resolves the path for the current model version.
- `finetune_directory`: Constructs fine-tuning directory names dynamically.
- `lstm_static_input_size`: Computes the input size for static features. Will be used in constructing the model.
- `lstm_dynamic_input_feature_size`: Computes the input size for dynamic features. Will be used in constructing the model.

#### **Key Methods**

1. **`update(kwargs: Dict[str, Any])`:**
   Updates the configuration with new key-value pairs.

2. **`save_to_yaml(filepath: str)`:**
   Saves the current configuration to a YAML file.

3. **`load_from_yaml(filepath: str)`:**
   Loads configuration parameters from a YAML file.

4. **`set_new_version_name()`:**
   Generates a new version name for managing model outputs.

5. **`__repr__()`:**
   Provides a string representation of the configuration.

6. **`get(*args: str)`:**
   Retrieves the value of a configuration parameter by key.

7. **`make_dirpath()`:**
   Creates directory paths for saving model artifacts.

---

## Example Usage

### Loading a Configuration

```python
from hydroml.config import load_config

config_path = "config.yaml"
config = load_config(config_path)
print(config)
```

### Updating Configuration

```python
config.update({'lr': 5e-4, 'batch_size': 256})
```

### Saving Configuration

```python
config.save_to_yaml("updated_config.yaml")
```

### Accessing Dynamic Properties

```python
print(f"LSTM Static Input Size: {config.lstm_static_input_size}")
print(f"Parent Path: {config.parent_path}")
```


## Configuration Parameters

### **Model Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| model                                        | 'HydroLSTM'                  | Model architecture to use            |
| lstm_dynamic_input_feature_latent_size      | 1                            | Latent size for dynamic input features|
| lstm_static_input_latent_size               | 1                            | Latent size for static input features |
| lstm_hidden_size                            | 256                          | Hidden layer size of LSTM            |
| lstm_target_size                            | 1                            | Output size of LSTM                  |
| lstm_layers                                 | 1                            | Number of LSTM layers                |
| dropout_probability                         | 0.0                          | Dropout probability                  |
| initial_forget_bias                         | 0.5                          | Initial bias for forget gate         |

### **Training Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| lr                                           | 1e-4                          | Learning rate                        |
| weight_decay                                 | 0.0001                        | L2 regularization factor             |
| loss_fn                                      | 'nse'                         | Loss function                        |
| batch_size                                   | 512                           | Batch size for training              |
| max_epochs                                   | 120                           | Maximum training epochs              |
| check_val_every_n_epoch                     | 5                             | Validation check frequency            |
| gradient_clip_val                           | 0.0                           | Gradient clipping value              |
| enable_progress_bar                         | True                          | Show progress bar during training    |

### **Feature Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| dynamic_features                             | ['precipitation_AWAP', 'et_morton_wet_SILO'] | Dynamic input features              |
| target_features                              | ['streamflow_mmd']           | Target features to predict           |
| static_features                              | ['catchment_area', 'mean_slope_pct', ...] | Static catchment features           |
| evolving_static_features                     | {'dynamic_feature_mean': {...}} | Time-evolving static features       |
| add_sin_cos_doy                              | True                         | Add sine/cosine day of year features |
| seqlen                                       | 365                          | Sequence length for input            |
| transform_predictor_dynamic                  | 'norm'                       | Transformation for dynamic predictors|
| transform_target                             | 'norm'                       | Transformation for target variables  |

### **Dataset Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| dataset_config.name                          | 'camels_aus_v1'              | Dataset name                         |
| dataset_config.static_feature_path_key      | 'camels_aus_attributes'      | Path key for static features         |
| dataset_config.dir_key                       | 'camels_aus_v1'              | Directory key for dataset            |

### **Hardware Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| device                                       | 'cpu'                        | Computing device (cpu/cuda)          |
| dataloader_nworkers                         | 12                           | Number of dataloader workers         |
| dataloader_persistent_workers               | True                         | Keep workers alive between epochs    |

### **Finetune Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| save_top_k                                   | 1                            | Number of best models to save        |
| layers_to_finetune                           | None                         | Specific layers to finetune          |
| finetune_max_epochs                         | 15                           | Maximum epochs for finetuning        |
| finetune_lr                                  | 7e-5                          | Learning rate for finetuning         |

### **Path Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| parent_path_key                              | 'parent_path'                | Key for parent directory path        |
| transform_parameter_path                    | 'params.yaml'                | Path for transformation parameters  |
| finetune_directory_name_base                | 'finetune'                   | Base name for finetune directory     |
| name                                         | 'default'                    | Configuration name                   |
| version                                      | 'none'                       | Version identifier                   |

### **Miscellaneous Configuration**
| Parameter                                    | Default Value                | Description                           |
|----------------------------------------------|------------------------------|---------------------------------------|
| number_of_time_output_timestep              | 1                            | Number of output timesteps           |
| is_train                                     | True                         | Training mode flag                   |
| head_hidden_layers                           | [10]                         | Hidden layer sizes in head network   |

