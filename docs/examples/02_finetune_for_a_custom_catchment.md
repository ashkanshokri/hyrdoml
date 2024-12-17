```python
# imports
%load_ext autoreload 
%autoreload 2
 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from hydroml.utils import helpers as h
from hydroml.training.finetune import run_finetune_from_timeseries
from hydroml.workflow.prediction import run_hydrological_simulation
from hydroml.evaluation.metrics import Metrics   
from hydroml.config.config import load_config

def get_metrics(ds):
    metrics = Metrics(ds['y'], ds['prediction']).all_metrics().to_dataframe().reset_index().drop(columns=['catchment_id', 'lead_time', 'feature'])
    return metrics
```


```python
# For this example we need a trained model.
model_path = Path('../sample_data/model/version_0')

# we need to convert the transform_parameter_path to an absolute path so all the finetuned models
# to refer to the same parameters and do not calculate a new one for each catchment.
transform_parameter_path = (model_path / 'params.yaml').absolute()


catchment_id = '401208'
dynamic_data=pd.read_csv(f'../sample_data/{catchment_id}.csv', index_col=0, parse_dates=True)
static_data=h.read_json(f'../sample_data/{catchment_id}_attributes.json')

```

# Split the data into calibration and validation periods
We extract the calibration and validation periods from the config file and use them to split our data.
This ensures we use the same periods that were used during model training.



```python
from hydroml.config.config import load_config


config = load_config(model_path / 'config.yaml')
cal_periods = config.cal['periods']
val_periods = config.val['periods']

cal_dynamic_data = pd.concat([dynamic_data.loc[s:e] for s, e in cal_periods])
val_dynamic_data = pd.concat([dynamic_data.loc[s:e] for s, e in val_periods])
```

# Run the simulation
For benchmarking we run the original/pretrained model first.

We can easily adjust the config parameters for the simulation by passing them as kwargs to the run_hydrological_simulation function here we need to change the device to cpu and pass the transform_parameter_path to the simulation so it uses the same parameters as the finetuned model.


```python
kwargs = {'transform_parameter_path': transform_parameter_path,
          'device': 'cpu', 
          }
```


```python
simulation_using_original_model = run_hydrological_simulation(model_path, val_dynamic_data, static_data, catchment_id, **kwargs)
```

    Transforming data: loading transform parameters from l:\work\sho108_handover\hydroml\examples\notebooks\..\sample_data\model\version_0\params.yaml
    

    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

Now we finetune the model for the catchment we are interested in using the calibration data. Then we can run the simulation using the finetuned model for the validation data.


```python
# We need to adjust batch size to be able to fit the model in the memory. device='cpu' if no gpu is available.
# When no layer_to_finetune is provided, all paarameters in the model are tuned.
p,v = run_finetune_from_timeseries(model_path, cal_dynamic_data, static_data, catchment_id, device='cpu', batch_size=128)
finetuned_model_path = Path(p) / v
simulation_using_finetuned_model = run_hydrological_simulation(finetuned_model_path, val_dynamic_data, static_data, catchment_id, **kwargs)


metrics =pd.concat([get_metrics(simulation_using_original_model), get_metrics(simulation_using_finetuned_model)]).T
metrics.columns = ['original', 'finetuned']

metrics
```

    params.yaml
    Transforming data: loading transform parameters from ..\sample_data\model\version_0\params.yaml
    

    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    ..\sample_data\model\version_0\finetune_all\401208\241218091130_012d
    

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\trainer\configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\hydroml\examples\sample_data\model\version_0\finetune_all\401208\241218091130_012d exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 6      | train
    2 | lstm              | LSTM       | 266 K  | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 2.6 K  | train
    ---------------------------------------------------------
    268 K     Trainable params
    0         Non-trainable params
    268 K     Total params
    1.075     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode
    


    Training: |          | 0/? [00:00<?, ?it/s]


    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 128. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 98. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:384: `ModelCheckpoint(monitor='val_loss')` could not find the monitored key in the returned metrics: ['lr-Adam', 'train_loss', 'epoch', 'step']. HINT: Did you call `log('val_loss', value)` in the `LightningModule`?
    `Trainer.fit` stopped: `max_epochs=15` reached.
    

    Transforming data: loading transform parameters from l:\work\sho108_handover\hydroml\examples\notebooks\..\sample_data\model\version_0\params.yaml
    

    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original</th>
      <th>finetuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nse</th>
      <td>0.662218</td>
      <td>0.890566</td>
    </tr>
    <tr>
      <th>kge</th>
      <td>0.602030</td>
      <td>0.829118</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.541452</td>
      <td>0.308190</td>
    </tr>
    <tr>
      <th>bias</th>
      <td>1.320835</td>
      <td>0.866343</td>
    </tr>
    <tr>
      <th>relative_bias</th>
      <td>0.320835</td>
      <td>-0.133657</td>
    </tr>
    <tr>
      <th>absolute_bias</th>
      <td>1.320835</td>
      <td>0.866343</td>
    </tr>
    <tr>
      <th>nse_sqrt</th>
      <td>0.720300</td>
      <td>0.785461</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we need to adjust batch size to be able to fit the model in the memory. device='cpu' if no gpu is available. 
# This tune only the parameters in the layers_to_finetune.
p,v = run_finetune_from_timeseries(model_path, cal_dynamic_data, static_data, catchment_id, device='cpu', batch_size=128, max_epochs=20, layers_to_finetune=['head', 'dynamic_embedding'])
partial_finetuned_model_path = Path(p) / v
simulation_using_partial_finetuned_model = run_hydrological_simulation(partial_finetuned_model_path, val_dynamic_data, static_data, catchment_id, **kwargs)




```

    params.yaml
    Transforming data: loading transform parameters from ..\sample_data\model\version_0\params.yaml
    

    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    ..\sample_data\model\version_0\finetune_head_dynamic_embedding\401208\241218092856_2651
    

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\trainer\configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\hydroml\examples\sample_data\model\version_0\finetune_head_dynamic_embedding\401208\241218092856_2651 exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 6      | train
    2 | lstm              | LSTM       | 266 K  | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 2.6 K  | train
    ---------------------------------------------------------
    2.6 K     Trainable params
    266 K     Non-trainable params
    268 K     Total params
    1.075     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode
    


    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=15` reached.
    

    Transforming data: loading transform parameters from l:\work\sho108_handover\hydroml\examples\notebooks\..\sample_data\model\version_0\params.yaml
    

    l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

# Calculate the metrics

To compare the performance of the different models we calculate the metrics for each model.


```python
metrics =pd.concat([get_metrics(simulation_using_original_model), get_metrics(simulation_using_finetuned_model), get_metrics(simulation_using_partial_finetuned_model)]).T
metrics.columns = ['original', 'finetuned', 'partial_finetuned']

metrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original</th>
      <th>finetuned</th>
      <th>partial_finetuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nse</th>
      <td>0.662218</td>
      <td>0.890566</td>
      <td>0.827960</td>
    </tr>
    <tr>
      <th>kge</th>
      <td>0.602030</td>
      <td>0.829118</td>
      <td>0.888905</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.541452</td>
      <td>0.308190</td>
      <td>0.386417</td>
    </tr>
    <tr>
      <th>bias</th>
      <td>1.320835</td>
      <td>0.866343</td>
      <td>1.054681</td>
    </tr>
    <tr>
      <th>relative_bias</th>
      <td>0.320835</td>
      <td>-0.133657</td>
      <td>0.054681</td>
    </tr>
    <tr>
      <th>absolute_bias</th>
      <td>1.320835</td>
      <td>0.866343</td>
      <td>1.054681</td>
    </tr>
    <tr>
      <th>nse_sqrt</th>
      <td>0.720300</td>
      <td>0.785461</td>
      <td>0.839433</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
