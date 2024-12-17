# Visualizing Evaluation Results

This notebook demonstrates how to visualize and compare results from different model experiments. We'll look at:

1. Loading simulation results from finetuned and continental models
2. Comparing metrics across different model versions
3. Visualizing the differences using various plots

The notebook uses simulation outputs from two model versions (old and new) in both finetuned and continental configurations.



```python
# Import required libraries 
from pathlib import Path
import xarray as xr
from hydroml.evaluation import graphs as g
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydroml.evaluation.metrics import get_metrics

# Function to read simulation results from finetuned models
# This reads multiple simulation.nc files from individual catchment directories
# and concatenates them into a single dataset
def read_simulation_of_finetuned_model(p, finetune_directory = 'finetune_all', leave=False):
    ds_list = []
    for path in tqdm(list((Path(p)/finetune_directory).glob('*/*/results/simulation.nc')), desc='Reading finetuned simulation', leave=leave):
        ds = xr.open_dataset(path)
        ds_list.append(ds)

    return xr.concat(ds_list, dim='catchment_id')


# Function to read simulation results from continental models
# This reads a single simulation.nc file containing results for all catchments
def read_simulation_of_continental_model(p):
    return xr.open_dataset(Path(p) / 'results' / 'simulation.nc')
```

# Comparing Model Performance

Now that we have loaded our simulation results, let's analyze and visualize the performance metrics across different model configurations.

We'll focus on:
- NSE (Nash-Sutcliffe Efficiency) scores for each catchment
- Distribution of performance metrics using exceedance curves
- Comparing performance between finetuned and continental models

 First, we'll set up the paths to our model experiments and load the simulation results.



```python
# this is the path to the models you want to compare

from hydroml.utils import config_path_handler as cp
from pathlib import Path
path_mapping = cp.get_path_mapping_file_from_config(platform='win_2')
path_to_all_models = Path(path_mapping['parent_path'])
```


```python
# select one trial of experiment toos_qc_validation (the lstm with climate and awra forcing), and toos_c_validation  (lstm with only climate forcing)

toos_qc_validation_path = next((path_to_all_models / 'toos_qc_validation').glob('*'))
toos_c_validation_path = next((path_to_all_models / 'toos_c_validation').glob('*'))


experiment_path_mapping = { 
    'toos_qc_validation': toos_qc_validation_path,
    'toos_c_validation': toos_c_validation_path,
}
```




    WindowsPath('//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/models/toos_qc_validation/241216120325_5de9')




```python

```


```python
# read simulation results from finetuned models
finetuned_simulations = {f'{k}_finetuned':read_simulation_of_finetuned_model(v) for k, v in experiment_path_mapping.items()}
continental_simulation = {f'{k}_continental':read_simulation_of_continental_model(v) for k, v in experiment_path_mapping.items()}

all_simulations = {**finetuned_simulations, **continental_simulation}

```

                                                    


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    File l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\xarray\core\concat.py:254, in concat(objs, dim, data_vars, coords, compat, positions, fill_value, join, combine_attrs, create_index_for_new_dim)
        253 try:
    --> 254     first_obj, objs = utils.peek_at(objs)
        255 except StopIteration as err:
    

    File l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\xarray\core\utils.py:199, in peek_at(iterable)
        198 gen = iter(iterable)
    --> 199 peek = next(gen)
        200 return peek, itertools.chain([peek], gen)
    

    StopIteration: 

    
    The above exception was the direct cause of the following exception:
    

    ValueError                                Traceback (most recent call last)

    Cell In[3], line 2
          1 # read simulation results from finetuned models
    ----> 2 finetuned_simulations = {f'{k}_finetuned':read_simulation_of_finetuned_model(v) for k, v in experiment_path_mapping.items()}
          3 continental_simulation = {f'{k}_continental':read_simulation_of_continental_model(v) for k, v in experiment_path_mapping.items()}
          5 all_simulations = {**finetuned_simulations, **continental_simulation}
    

    Cell In[3], line 2, in <dictcomp>(.0)
          1 # read simulation results from finetuned models
    ----> 2 finetuned_simulations = {f'{k}_finetuned':read_simulation_of_finetuned_model(v) for k, v in experiment_path_mapping.items()}
          3 continental_simulation = {f'{k}_continental':read_simulation_of_continental_model(v) for k, v in experiment_path_mapping.items()}
          5 all_simulations = {**finetuned_simulations, **continental_simulation}
    

    Cell In[1], line 18, in read_simulation_of_finetuned_model(p, finetune_directory, leave)
         15     ds = xr.open_dataset(path)
         16     ds_list.append(ds)
    ---> 18 return xr.concat(ds_list, dim='catchment_id')
    

    File l:\work\sho108_handover\hydroml\.venv\Lib\site-packages\xarray\core\concat.py:256, in concat(objs, dim, data_vars, coords, compat, positions, fill_value, join, combine_attrs, create_index_for_new_dim)
        254     first_obj, objs = utils.peek_at(objs)
        255 except StopIteration as err:
    --> 256     raise ValueError("must supply at least one object to concatenate") from err
        258 if compat not in set(_VALID_COMPAT) - {"minimal"}:
        259     raise ValueError(
        260         f"compat={compat!r} invalid: must be 'broadcast_equals', 'equals', 'identical', 'no_conflicts' or 'override'"
        261     )
    

    ValueError: must supply at least one object to concatenate



```python
# concatenate all simulation results into a single dataset
ds = xr.concat(all_simulations.values(), dim='experiments').assign_coords(experiments=list(all_simulations.keys())).squeeze()

```

## Performance Metrics Analysis

In this section, we calculate and analyze performance metrics across all catchments and model experiments.

The `get_metrics()` function computes several hydrological evaluation metrics including:
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE) 
- Other relevant metrics

Input Dataset Structure:
- **Experiments**: Contains both finetuned and continental model results
- **Variables**:
  - Simulated streamflow (model predictions)
  - Observed streamflow (ground truth)
  - Additional metadata

The metrics will help us quantitatively compare model performance across different configurations and catchments.


```python

metrics = get_metrics(ds)

# now we can inspect the metrics, here we look at NSE
metrics.to_dataframe()['nse'].unstack()

```

    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\numpy\lib\nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\numpy\lib\nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\numpy\lib\nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\numpy\lib\nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
      var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
    c:\Users\sho108\AppData\Local\pypoetry\Cache\virtualenvs\hydroml-dFLAodHf-py3.11\Lib\site-packages\xarray\core\computation.py:818: RuntimeWarning: invalid value encountered in sqrt
      result_data = func(*input_data)
    




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
      <th>experiments</th>
      <th>qc_1_finetuned</th>
      <th>qc_2_finetuned</th>
      <th>c_1_finetuned</th>
      <th>qc_1_continental</th>
      <th>qc_2_continental</th>
      <th>c_1_continental</th>
    </tr>
    <tr>
      <th>catchment_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102101A</th>
      <td>0.287034</td>
      <td>-0.366527</td>
      <td>-0.005387</td>
      <td>0.399578</td>
      <td>0.367988</td>
      <td>0.477361</td>
    </tr>
    <tr>
      <th>104001A</th>
      <td>0.626392</td>
      <td>0.607481</td>
      <td>0.578017</td>
      <td>0.621222</td>
      <td>0.462170</td>
      <td>0.564719</td>
    </tr>
    <tr>
      <th>105101A</th>
      <td>0.590319</td>
      <td>0.632638</td>
      <td>0.688484</td>
      <td>-0.924713</td>
      <td>-0.895629</td>
      <td>-1.044382</td>
    </tr>
    <tr>
      <th>105102A</th>
      <td>0.142363</td>
      <td>-0.115327</td>
      <td>-0.601462</td>
      <td>0.036103</td>
      <td>0.002300</td>
      <td>-0.042535</td>
    </tr>
    <tr>
      <th>105105A</th>
      <td>0.404815</td>
      <td>0.588827</td>
      <td>0.671565</td>
      <td>0.093816</td>
      <td>0.484427</td>
      <td>0.204214</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>G8200045</th>
      <td>0.520861</td>
      <td>0.518840</td>
      <td>0.548729</td>
      <td>0.523431</td>
      <td>0.494013</td>
      <td>0.609539</td>
    </tr>
    <tr>
      <th>G8210010</th>
      <td>0.525265</td>
      <td>0.373608</td>
      <td>0.402109</td>
      <td>0.535093</td>
      <td>0.568934</td>
      <td>0.576974</td>
    </tr>
    <tr>
      <th>G9030124</th>
      <td>0.439573</td>
      <td>0.400464</td>
      <td>0.418172</td>
      <td>0.281013</td>
      <td>0.325879</td>
      <td>0.028544</td>
    </tr>
    <tr>
      <th>G9030250</th>
      <td>0.360421</td>
      <td>0.744440</td>
      <td>0.803957</td>
      <td>0.572334</td>
      <td>0.600388</td>
      <td>0.715155</td>
    </tr>
    <tr>
      <th>G9070142</th>
      <td>0.441414</td>
      <td>0.829260</td>
      <td>0.878386</td>
      <td>0.391891</td>
      <td>0.593921</td>
      <td>0.805861</td>
    </tr>
  </tbody>
</table>
<p>219 rows × 6 columns</p>
</div>



## Exceedance Probability Analysis

Next, we'll analyze the distribution of NSE scores across catchments using exceedance probability curves.

An exceedance probability curve shows:
- The probability that a given NSE value will be exceeded
- Distribution and spread of model performance across catchments
- Comparison between different model configurations

We'll focus on catchments where all models produced valid results (no NaN values) for fair comparison.



```python
# to make sure we are comparing same catchments, we drop catchments with nan - this should not be the case normally, but in case some of the finetuneing is failed, we can still compare the catchments with non-nan NSE

non_nan_metrics = metrics.dropna(dim='catchment_id')

for experiment in ds.experiments.values:
    g.exceedance_curve(non_nan_metrics.sel(experiments=experiment)['nse'])


plt.title(f'NSE - {len(non_nan_metrics.catchment_id)} catchments')
plt.ylim(0, 1)
plt.legend(ds.experiments.values)
plt.grid(True, which='major', linestyle='-', alpha=0.9)
plt.grid(True, which='minor', linestyle='--', alpha=0.7)
plt.minorticks_on()
plt.show()

```


    
![png](05_visualization_files/05_visualization_11_0.png)
    



```python

```
