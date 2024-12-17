# HydroML



HydroML is an advanced Python package designed to enhance hydrological modeling through the application of Long Short-Term Memory (LSTM) networks. By leveraging the capabilities of LSTMs, HydroML addresses the challenges associated with streamflow prediction, offering a powerful tool for researchers and professionals in hydrology. Developed on the PyTorch Lightning framework, the package facilitates deep learning applications while supporting the integration of both static and dynamic catchment features. This dual functionality allows users to capture the complex interactions within hydrological systems, resulting in more accurate and reliable predictions.

A key feature of HydroML is its customizable configuration system, which enables users to adjust model architectures and training parameters to suit specific needs.  HydroML includes fine-tuning capabilities that allow users to adapt pre-trained models efficiently, making it possible to tailor models to new regions or specific catchment characteristics. The combination of flexibility and optimization makes HydroML a valuable tool for hydrological modeling, supporting improved outcomes in both research and practical applications.

In this readme, we will provide a basic introduction to the package and how to use it. For more detailed information, please refer to the please see:
- [Models and training](docs/models.md)
- [Configuration](docs/configuration.md)
- [Data Structure](docs/data_structure.md)


A more detailed evaluation is being prepared as a journal paper and will be submitted soon. To get an access to the evaluation results, please contact Ashkan Shokri (ash.shokri@csiro.au).


## Quick Start

### Install the package:

To install the package, you can use the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See our detailed [Installation Guide](docs/installation.md) for complete instructions.

### Prepare the path mapping (for a complete working example, see our [Path Mapping Notebook](examples/notebooks/03_build_a_path_mapping.ipynb)).

To use the package, you need to prepare a path mapping file which is a JSON file that contains the paths to the data and the parent directory where the model will be stored. The package can contain several different path mappings so for each platform we can use an appropriate path mapping file. The path mappings are stored in the config/path_mappings folder as JSON files. For this tutorial the path mapping needs to include 4 paths:
 
 - parent_path: The path to the parent directory where the model will be stored
 - camels_aus_attributes: The path to the CAMELS Australia attributes file
 - camels_aus_v1: The path to the CAMELS Australia v1 data
 - awra_postprocessed: The path to the AWRA postprocessed data
 
 In the next cell we are going to build a path mapping and save it as a JSON file in the config/path_mappings folder.

```python
path_mapping = {
    "parent_path": "P://work//sho108//hydroml//results",
    "camels_aus_attributes": "Z://Data//CAMELS_AUS//CAMELS_AUS_Attributes&Indices_MasterTable.csv",
    "camels_aus_v1": "Z://Data//CAMELS_AUS//preprocessed", 
    "awra_postprocessed": "L://work//sho108//AWRA//historical//v1//AWRALv7//preprocessed_catchment_mean//qtot_3"
}

from hydroml.utils import helpers as h
from hydroml.utils import config_path_handler as cp

platform_name = 'win_2' # this can be any arbitrary name. So you can refer to the path mapping later.

path_mapping_file_path = cp.get_path_mapping_file(platform_name)

h.save_json(path_mapping, path_mapping_file_path)

```


### Simulate runoff using climate data (for a complete working example, see our [Simulation Notebook](examples/notebooks/01_simulation.ipynb)).

There are different ways to use the package to simulate the runoff. A basic usage of the package to simulate the runoff is as follows:

To do this, you need a pre-trained model. See [Training Guide](docs/training.md) for details. A list of pre-trained models can be found in the [Models](docs/models.md) section. Then you can simply pass a set of dynamic and static data to the `run_hydrological_simulation` function as follows:


```python
import pandas as pd
from hydroml.workflow.prediction import run_hydrological_simulation

# Load your catchment data
dynamic_data = pd.read_csv('catchment_timeseries.csv', parse_dates=['date'], index_col='date')
static_data = {
    'catchment_area': 234.5,
    'mean_slope_pct': 12.3,
    'prop_forested': 0.45
}

# Load pre-trained model
model_path = 'path/to/pretrained/model'

# Run simulation
simulation = run_hydrological_simulation(model_path, dynamic_data, static_data, catchment_id, device='cpu')
```

Your "dynamic_data" should be formatted as:

```csv
date,precipitation_AWAP,et_morton_wet_SILO,qtot,streamflow_mmd
2010-01-01,1.2,3.4,0.5,0.1
2010-01-02,0.0,3.2,0.4,0.2
...
```

and "static_data" should be a dictionary with the keys that the model was trained on.


See [Data Structure](docs/data_structure.md) for detailed data format requirements.



# Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| 01 Simulation | Learn how to use pre-trained models to generate runoff predictions from climate data for any catchment. Step-by-step guide with example data and code. | [Markdown version](docs/examples/01_simulation.md) \| [Notebook](examples/notebooks/01_simulation.ipynb) |
| 02 Finetune for a custom catchment | Adapt existing models to your specific catchment by fine-tuning with local data. Includes tips for handling different data formats and optimizing model performance. | [Markdown version](docs/examples/02_finetune_for_a_custom_catchment.md) \| [Notebook](examples/notebooks/02_finetune_for_a_custom_catchment.ipynb) |
| 03 Build a path mapping | Learn how to set up path mappings to organize your data and model files. Essential for working with multiple data sources and model versions. | [Markdown version](docs/examples/03_build_a_path_mapping.md) \| [Notebook](examples/notebooks/03_build_a_path_mapping.ipynb) |
| 04 Run evaluation | Comprehensive guide to evaluating model performance using various metrics. Learn how to assess prediction accuracy and model reliability. | [Markdown version](docs/examples/04_run_evaluation.md) \| [Notebook](examples/notebooks/04_run_evaluation.ipynb) |
| 05 Visualization | Master the visualization tools to create insightful plots and graphs of your simulation results. Includes time series plots, error analysis, and performance metrics. | [Markdown version](docs/examples/05_visualization.md) \| [Notebook](examples/notebooks/05_visualization.ipynb) |
