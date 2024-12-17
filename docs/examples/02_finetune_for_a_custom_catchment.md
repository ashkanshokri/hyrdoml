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


def get_metrics(ds):
    metrics = Metrics(ds['y'], ds['prediction']).all_metrics().to_dataframe().reset_index().drop(columns=['catchment_id', 'lead_time', 'feature'])
    return metrics
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
# For this example we need a trained model.
model_path = Path('sample_data/model/version_0')
transform_parameter_path = (model_path.parent / 'params.yaml').absolute()


catchment_id = '401208'
dynamic_data=pd.read_csv(f'sample_data/{catchment_id}.csv', index_col=0, parse_dates=True)
static_data=h.read_json(f'sample_data/{catchment_id}_attributes.json')

val_dynamic_data = dynamic_data['1975': '2000']
cal_dynamic_data = dynamic_data['2001': '2014']
```


```python
simulation_using_original_model = run_hydrological_simulation(model_path, val_dynamic_data, static_data, catchment_id, device='cpu', transform_parameter_path=transform_parameter_path)


# we need to adjust batch size to be able to fit the model in the memory. device='cpu' if no gpu is available. This tune all paarameters in the model
p,v = run_finetune_from_timeseries(model_path, cal_dynamic_data, static_data, catchment_id, device='cpu', batch_size=128)
finetuned_model_path = Path(p) / v
simulation_using_finetuned_model = run_hydrological_simulation(finetuned_model_path, val_dynamic_data, static_data, catchment_id, device='cpu', transform_parameter_path=transform_parameter_path)


metrics =pd.concat([get_metrics(simulation_using_original_model), get_metrics(simulation_using_finetuned_model)]).T
metrics.columns = ['original', 'finetuned']

metrics
```

    Transforming data: loading transform parameters from p:\work\sho108\hydroml\examples\notebooks\sample_data\model\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(


    params.yaml
    Transforming data: loading transform parameters from sample_data\model\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs


    sample_data\model\version_0\finetune_all\401208\241212183016_6b6c


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\examples\notebooks\sample_data\model\version_0\finetune_all\401208\241212183016_6b6c exists and is not empty.
    
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
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:298: The number of training batches (38) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 128. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 11. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:384: `ModelCheckpoint(monitor='val_loss')` could not find the monitored key in the returned metrics: ['lr-Adam', 'train_loss', 'epoch', 'step']. HINT: Did you call `log('val_loss', value)` in the `LightningModule`?
    `Trainer.fit` stopped: `max_epochs=5` reached.


    Transforming data: loading transform parameters from p:\work\sho108\hydroml\examples\notebooks\sample_data\model\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
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
      <td>0.622850</td>
      <td>0.847155</td>
    </tr>
    <tr>
      <th>kge</th>
      <td>0.637977</td>
      <td>0.833720</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.579052</td>
      <td>0.368626</td>
    </tr>
    <tr>
      <th>bias</th>
      <td>1.039464</td>
      <td>0.970923</td>
    </tr>
    <tr>
      <th>relative_bias</th>
      <td>0.039464</td>
      <td>-0.029077</td>
    </tr>
    <tr>
      <th>absolute_bias</th>
      <td>1.039464</td>
      <td>0.970923</td>
    </tr>
    <tr>
      <th>nse_sqrt</th>
      <td>0.584891</td>
      <td>0.881859</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we need to adjust batch size to be able to fit the model in the memory. device='cpu' if no gpu is available. This tune all paarameters in the model
p,v = run_finetune_from_timeseries(model_path, cal_dynamic_data, static_data, catchment_id, device='cpu', batch_size=128, layers_to_finetune=['head', 'dynamic_embedding'])
partial_finetuned_model_path = Path(p) / v
simulation_using_partial_finetuned_model = run_hydrological_simulation(partial_finetuned_model_path, val_dynamic_data, static_data, catchment_id, device='cpu', transform_parameter_path=transform_parameter_path)

metrics =pd.concat([get_metrics(simulation_using_original_model), get_metrics(simulation_using_finetuned_model), get_metrics(simulation_using_partial_finetuned_model)]).T
metrics.columns = ['original', 'finetuned', 'partial_finetuned']

metrics


```

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs


    params.yaml
    Transforming data: loading transform parameters from sample_data\model\params.yaml
    sample_data\model\version_0\finetune_head_dynamic_embedding\401208\241212183214_22db


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\trainer\configuration_validator.py:70: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{ev-ca-macq}\work\sho108\hydroml\examples\notebooks\sample_data\model\version_0\finetune_head_dynamic_embedding\401208\241212183214_22db exists and is not empty.
    
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
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:298: The number of training batches (38) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=5` reached.


    Transforming data: loading transform parameters from p:\work\sho108\hydroml\examples\notebooks\sample_data\model\params.yaml


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
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
      <th>partial_finetuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nse</th>
      <td>0.622850</td>
      <td>0.847155</td>
      <td>0.739089</td>
    </tr>
    <tr>
      <th>kge</th>
      <td>0.637977</td>
      <td>0.833720</td>
      <td>0.714330</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>0.579052</td>
      <td>0.368626</td>
      <td>0.481622</td>
    </tr>
    <tr>
      <th>bias</th>
      <td>1.039464</td>
      <td>0.970923</td>
      <td>0.738604</td>
    </tr>
    <tr>
      <th>relative_bias</th>
      <td>0.039464</td>
      <td>-0.029077</td>
      <td>-0.261396</td>
    </tr>
    <tr>
      <th>absolute_bias</th>
      <td>1.039464</td>
      <td>0.970923</td>
      <td>0.738604</td>
    </tr>
    <tr>
      <th>nse_sqrt</th>
      <td>0.584891</td>
      <td>0.881859</td>
      <td>0.578552</td>
    </tr>
  </tbody>
</table>
</div>


