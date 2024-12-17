# Temporal Out-of-Sample Validation Tutorial

In this tutorial, we'll demonstrate how to perform temporal out-of-sample validation using the full CAMELS-Australia dataset instead of sample data. We'll leverage the built-in tools from the `hydroml` package to automatically handle data loading and execute the training and evaluation pipeline.

The workflow will:
1. Load the CAMELS-Australia dataset
2. Set up temporal splits for calibration and validation
3. Train and fine-tune a model
4. Evaluate model performance

Let's get started!



```python
from hydroml.config.config import Config
from hydroml.workflow.evaluation import train_finetune_evaluate
```

# Set up the path mapping

As this training is going to load the dataset automativally from camels australia, we need to set up the path mapping first. The method to set up the path mapping is the same as the one in the 03_build_a_path_mapping.ipynb tutorial.



```python
# We assume that the path mapping is already set up with the platform name 'win_2'.
platform='win_2'
```

We also need a list of basins for calibration and validation. These catchments should be already available in the camels australia postprocessed dataset and awara postprocessed dataset. So the pipeline will automatically load the data.


```python
basins_file = '../sample_data/basins.txt'
with open(basins_file, 'r') as f:
    catchment_ids = f.read().splitlines()
```

Using these information we can set up the config object.

For a full explanation of the config object please refer to the readme documentation.


```python
# we only use 2 catchments for calibration and validation for this tutorial.
config = Config(cal={'periods' : [['1991-01-01', '2014-01-01']], 'catchment_ids':catchment_ids[:2] },  
                val={'periods' : [['1985-01-01', '1990-01-01']], 'catchment_ids':catchment_ids[:2] }, 
                name = 'evaluation_tutorial',
                lstm_hidden_size=4, # we are selecting a small lstm size for this tutorial.
                device='cpu',
                platform=platform, # to introduce the paths stored in config/path_mapping/win.yaml
                max_epochs=2, # and we reduce the number of epochs to 2 for this tutorial.
                )    

# we  need to call this function to set up the version name - or you can manually set it up.
config.set_new_version_name()
```


Now we can run the **training and evaluation pipeline**. This pipeline works as follows:

1. Trains a **continental model** and evaluates it on the validation set.  
2. Starts the **fine-tuning process**, where the continental model is fine-tuned for each catchment and then evaluated on that specific catchment.

The results will be stored in the following structure:

| **File/Folder**                          | **Description**                                                      | **Path**                                                          |
|------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| **Model Weights**                        | Weights of the trained continental model.                           | `root_path/VERSION_NAME/last.ckpt`                                |
| **Config File**                          | Configuration settings for the pipeline.                            | `root_path/VERSION_NAME/config.yaml`                              |
| **Transformer Parameters**               | Parameters for the transformer model.                               | `root_path/VERSION_NAME/params.yaml`                              |
| **Simulation Results (Continental Model)** | Simulation output for the continental model.                        | `root_path/VERSION_NAME/simulation.nc`                            |
| **Metrics**                              | Evaluation metrics for the continental model.                       | `root_path/VERSION_NAME/metrics.nc`                               |
| **Fine-tuned Models**                    | Fine-tuned models for each catchment.                               | `root_path/VERSION_NAME/finetune_all/catchment_id/VERSION_NAME/last.ckpt` |

- Note 1: that the finetuned models are stored with a same structure as the continental model.
- Note 2: the finetuned models do not have a params.yaml file as they are config to use the transformer parameter in the continental model.



```python
train_finetune_evaluate(config)
```

                                                                            

    Transforming data: calculating transform parameters and saving to \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    valid data points per catchment {0: 5381, 1: 2219}
    

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    GPU available: False, used: False
    

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1
    

    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1 exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 5      | train
    2 | lstm              | LSTM       | 128    | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 61     | train
    ---------------------------------------------------------
    209       Trainable params
    0         Non-trainable params
    209       Total params
    0.001     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 512. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:298: The number of training batches (15) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
    


    Training: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 432. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:384: `ModelCheckpoint(monitor='val_loss')` could not find the monitored key in the returned metrics: ['lr-Adam', 'train_loss', 'epoch', 'step']. HINT: Did you call `log('val_loss', value)` in the `LightningModule`?
    `Trainer.fit` stopped: `max_epochs=2` reached.
                                                                            

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
                                                                            

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\finetune_all\912101A\241218102312_4208 exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 5      | train
    2 | lstm              | LSTM       | 128    | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 61     | train
    ---------------------------------------------------------
    209       Trainable params
    0         Non-trainable params
    209       Total params
    0.001     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode
    

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\finetune_all\912101A\241218102312_4208
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:298: The number of training batches (11) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
    


    Training: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 261. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    


    Validation: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 72. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=15` reached.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    

                                                                            

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\finetune_all\912105A\241218102502_1497
    

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:654: Checkpoint directory \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\finetune_all\912105A\241218102502_1497 exists and is not empty.
    
      | Name              | Type       | Params | Mode 
    ---------------------------------------------------------
    0 | static_embedding  | Linear     | 15     | train
    1 | dynamic_embedding | Linear     | 5      | train
    2 | lstm              | LSTM       | 128    | train
    3 | dropout           | Identity   | 0      | train
    4 | head              | Sequential | 61     | train
    ---------------------------------------------------------
    209       Trainable params
    0         Non-trainable params
    209       Total params
    0.001     Total estimated model params size (MB)
    9         Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 296. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\loops\fit_loop.py:298: The number of training batches (5) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
    


    Training: |          | 0/? [00:00<?, ?it/s]


    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\pytorch_lightning\utilities\data.py:78: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 171. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=15` reached.
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-vTEL7SXK-py3.11\Lib\site-packages\torch\utils\data\dataloader.py:617: UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is 10 (`cpuset` is not taken into account), which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    

    Transforming data: loading transform parameters from \\fs1-cbr.nexus.csiro.au\{d61-coastal-forecasting-wp3}\work\sho108_handover\models\evaluation_tutorial\241218102129_2bd1\params.yaml
    




    WindowsPath('//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/models/evaluation_tutorial/241218102129_2bd1')



So the new set of results for the simulation over the validtion period using the continental model is stored in the following path:



```python
import xarray as xr

p = config.current_path / config.version /'results' / 'simulation.nc'
xr.open_dataset(p)

```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1f1f1f;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 35kB
Dimensions:       (date: 1462, catchment_id: 2, lead_time: 1, feature: 1)
Coordinates:
  * date          (date) datetime64[ns] 12kB 1986-01-01 ... 1990-01-01
  * catchment_id  (catchment_id) &lt;U7 56B &#x27;912101A&#x27; &#x27;912105A&#x27;
Dimensions without coordinates: lead_time, feature
Data variables:
    prediction    (lead_time, feature, date, catchment_id) float32 12kB ...
    y             (lead_time, feature, date, catchment_id) float32 12kB ...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-3d8044a5-1e5f-45a4-9bdf-fc7ea2179627' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3d8044a5-1e5f-45a4-9bdf-fc7ea2179627' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>date</span>: 1462</li><li><span class='xr-has-index'>catchment_id</span>: 2</li><li><span>lead_time</span>: 1</li><li><span>feature</span>: 1</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-da546c83-e4f8-4e4a-8b2d-8363c1ca9986' class='xr-section-summary-in' type='checkbox'  checked><label for='section-da546c83-e4f8-4e4a-8b2d-8363c1ca9986' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>date</span></div><div class='xr-var-dims'>(date)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>1986-01-01 ... 1990-01-01</div><input id='attrs-7f9f45b6-a682-4d26-a4ad-6c06c07c33ba' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7f9f45b6-a682-4d26-a4ad-6c06c07c33ba' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3267babd-667f-42b6-9578-c40c024fa27c' class='xr-var-data-in' type='checkbox'><label for='data-3267babd-667f-42b6-9578-c40c024fa27c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;1986-01-01T00:00:00.000000000&#x27;, &#x27;1986-01-02T00:00:00.000000000&#x27;,
       &#x27;1986-01-03T00:00:00.000000000&#x27;, ..., &#x27;1989-12-30T00:00:00.000000000&#x27;,
       &#x27;1989-12-31T00:00:00.000000000&#x27;, &#x27;1990-01-01T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>catchment_id</span></div><div class='xr-var-dims'>(catchment_id)</div><div class='xr-var-dtype'>&lt;U7</div><div class='xr-var-preview xr-preview'>&#x27;912101A&#x27; &#x27;912105A&#x27;</div><input id='attrs-8f7d8e94-94b6-48d5-97b1-ff9e2a003074' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f7d8e94-94b6-48d5-97b1-ff9e2a003074' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6885ef3a-d24a-4d47-9ffa-47687e11cd56' class='xr-var-data-in' type='checkbox'><label for='data-6885ef3a-d24a-4d47-9ffa-47687e11cd56' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;912101A&#x27;, &#x27;912105A&#x27;], dtype=&#x27;&lt;U7&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ae9c1745-32cb-40d0-9009-b270d0614d68' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ae9c1745-32cb-40d0-9009-b270d0614d68' class='xr-section-summary' >Data variables: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>prediction</span></div><div class='xr-var-dims'>(lead_time, feature, date, catchment_id)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-21b22b89-1675-401a-b94f-6e5b1d684dd2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21b22b89-1675-401a-b94f-6e5b1d684dd2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8ee1a430-9dc7-4837-a3d2-2fde05c0a9ce' class='xr-var-data-in' type='checkbox'><label for='data-8ee1a430-9dc7-4837-a3d2-2fde05c0a9ce' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[2924 values with dtype=float32]</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(lead_time, feature, date, catchment_id)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-f1f1aa95-eb88-442e-b27e-4b71520cc48b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f1f1aa95-eb88-442e-b27e-4b71520cc48b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5408377c-1358-4855-b2fa-8617ee25e30b' class='xr-var-data-in' type='checkbox'><label for='data-5408377c-1358-4855-b2fa-8617ee25e30b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>[2924 values with dtype=float32]</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d08e7fad-05ef-4aa3-b587-855cabc82e00' class='xr-section-summary-in' type='checkbox'  ><label for='section-d08e7fad-05ef-4aa3-b587-855cabc82e00' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>date</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-8591dbf4-88f9-4455-94d2-4df6f0c6bedc' class='xr-index-data-in' type='checkbox'/><label for='index-8591dbf4-88f9-4455-94d2-4df6f0c6bedc' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(DatetimeIndex([&#x27;1986-01-01&#x27;, &#x27;1986-01-02&#x27;, &#x27;1986-01-03&#x27;, &#x27;1986-01-04&#x27;,
               &#x27;1986-01-05&#x27;, &#x27;1986-01-06&#x27;, &#x27;1986-01-07&#x27;, &#x27;1986-01-08&#x27;,
               &#x27;1986-01-09&#x27;, &#x27;1986-01-10&#x27;,
               ...
               &#x27;1989-12-23&#x27;, &#x27;1989-12-24&#x27;, &#x27;1989-12-25&#x27;, &#x27;1989-12-26&#x27;,
               &#x27;1989-12-27&#x27;, &#x27;1989-12-28&#x27;, &#x27;1989-12-29&#x27;, &#x27;1989-12-30&#x27;,
               &#x27;1989-12-31&#x27;, &#x27;1990-01-01&#x27;],
              dtype=&#x27;datetime64[ns]&#x27;, name=&#x27;date&#x27;, length=1462, freq=None))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>catchment_id</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-278b85ae-6894-40a9-b169-48a172407baf' class='xr-index-data-in' type='checkbox'/><label for='index-278b85ae-6894-40a9-b169-48a172407baf' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;912101A&#x27;, &#x27;912105A&#x27;], dtype=&#x27;object&#x27;, name=&#x27;catchment_id&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e51bd401-92c8-495d-8560-b98e8b774f87' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-e51bd401-92c8-495d-8560-b98e8b774f87' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>


