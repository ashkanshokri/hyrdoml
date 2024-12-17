# Data Structure

In HydroML, the data is handled through the data module. The base dataset class is `hydroml.data.dataset.Dataset`.

## Dataset

The required format for the `xarray.Dataset` (referred to as `ds`) in HydroML is as follows:

### Dimensions:
- **catchment_id**: Represents the unique identifiers for each catchment. In this case, there are 2 catchments.
- **date**: The time dimension, covering a range of 3654 days (from `1975-01-01` to `1985-01-01`).
- **dynamic_feature**: The number of dynamic input features, with 25 features (e.g., precipitation, streamflow, temperature).
- **static_feature_dim**: The number of static features per catchment (13 features such as catchment area, slope, etc.).
- **metadata_dim**: The number of metadata dimensions, here it is 1 (e.g., catchment-specific metadata like weight).

### Coordinates:
- **date**: The coordinate representing dates, stored as `datetime64[ns]`. For example, spanning from `1975-01-01` to `1985-01-01` (3654 dates).
- **dynamic_feature**: A list of dynamic feature names, such as `'precipitation_AWAP'`, `'temperature_max'`, and `'qtot'`, representing time-varying input features.
- **catchment_id**: The unique catchment identifiers, represented as strings (e.g., `'912101A'`, `'912105A'`).
- **metadata_dim**: A list representing metadata dimensions, such as `'weight'`.

### Data Variables:
- **dynamic**: A 3D array of size `(catchment_id, date, dynamic_feature)` that stores the dynamic input data (e.g., climate data, streamflow data). The values are stored as `float64` and represent the data for each catchment, for each date, and for each dynamic feature.
  
  Example: `dynamic` data may represent time-varying climate and hydrological data for each catchment, such as precipitation, temperature, or streamflow.

- **static**: A 2D array of size `(catchment_id, static_feature_dim)` that holds static features for each catchment. These features remain constant over time (e.g., catchment area, average slope). These values are stored as `float64`.

- **metadata**: A 2D array of size `(catchment_id, metadata_dim)` that stores metadata associated with each catchment. For instance, it could contain the weight of each catchment used for weighting the loss function or model training. This is stored as `int32`.

### Example:

```plaintext
Dimensions:
  catchment_id      : 2
  date              : 3654
  dynamic_feature   : 25
  static_feature_dim: 13
  metadata_dim      : 1

Coordinates:
  date              : 1975-01-01 ... 1985-01-01 (3654 days)
  dynamic_feature   : 'precipitation_AWAP', 'temperature_max', ... 'qtot' (25 features)
  catchment_id      : '912101A', '912105A' (2 catchments)
  metadata_dim      : 'weight' (1 metadata dimension)

Data variables:
  dynamic           : (catchment_id, date, dynamic_feature) float64 14.77 13.64 ... 0.03633 4.122e-05 (dynamic feature values)
  static            : (catchment_id, static_feature_dim) float64 1.258e+04 3.06 ... 20.38 1.669 (static feature values)
  metadata          : (catchment_id, metadata_dim) int32 1 1 (metadata values)

Indexes: (4)

Attributes:
  weight            : 1 (metadata associated with each catchment)
```

### Key Notes:

- **Dynamic Data**: These are the time-varying input features and target features, usually representing climate data and streamflow data. 
  - If `add_sin_cos_doy` is set to `True`, the sine and cosine of day-of-year features will be added to the dynamic data as follows:

```python
def get_sin_cos_doy(date: xr.DataArray) -> torch.Tensor:
    doy = date.dt.dayofyear
    normalized_doy = doy / 365.0
    sin_doy = np.sin(2 * np.pi * normalized_doy)
    cos_doy = np.cos(2 * np.pi * normalized_doy)
    return np.column_stack((sin_doy, cos_doy))
```

- **Static Data**: These are the constant features of the catchments, which remain unchanged over time. They are used as predictors for the model and are repeated throughout the time series for each catchment.

- **Metadata**: These values are also constant for each catchment, but they are not used directly as predictors in the model. Instead, they are used in other parts of the pipeline. For example, to use the NSE (Nash-Sutcliffe Efficiency) score as a loss function, the variance of the target variable must be calculated beforehand. This variance is passed to the loss function along with the ground truth and predicted target. Another example is when a weight for each catchment is needed to control its influence on the training process.

- **Evolving Static Features**: While these features are still considered static in nature (i.e., their value remains constant when predicting for a specific date), their values may change over time. For example, the annual mean evapotranspiration of a catchment could be treated as static over a short period of time (e.g., during a simulation of a single day informed by the past 365 days). However, over longer periods, the annual mean evapotranspiration may vary even for the same catchment. To account for this, we introduce the concept of evolving static features. These features are computed using a function and can change over time. In the modeling pipeline, they are passed to the model in the same way as static features.

- **Evolving Metadata**: The concept is the same as evolving static features—these are metadata that change over time, and they are used in a similar way in the pipeline.


# Constructing a Dataset
There are multiple ways of constructing a dataset. The easiest way is to use the `hydroml.data.dataset.Dataset.from_catchments` method. This method takes a list of `Catchment` objects and constructs a dataset from them. For more details, please refer to the [Catchment](catchment.md) documentation.

# Example: Constructing a Dataset using a List of Catchments

In this example, we will create multiple `Catchment` objects, convert them to `xarray.Dataset` objects, and combine them into a single `xarray.Dataset` for further analysis.

### Step 1: Create Catchment Objects

```python
import pandas as pd
from your_module import Catchment  # Adjust the import based on your module's location

# Sample dynamic data (time-series)
dynamic_data_1 = pd.DataFrame({
    'date': pd.date_range('2000-01-01', periods=5),
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10]
})
dynamic_data_1.set_index('date', inplace=True)

static_data_1 = ['land_area_1', 'vegetation_type_1']
metadata_1 = {'catchment_name': 'Catchment 1', 'location': 'Region A'}
catchment_id_1 = 'C001'
catchment_1 = Catchment(dynamic_data_1, static_data_1, metadata_1, catchment_id_1)

# Sample dynamic data (time-series)
dynamic_data_2 = pd.DataFrame({
    'date': pd.date_range('2000-01-01', periods=5),
    'feature1': [2, 3, 4, 5, 6],
    'feature2': [7, 8, 9, 10, 11]
})
dynamic_data_2.set_index('date', inplace=True)

static_data_2 = ['land_area_2', 'vegetation_type_2']
metadata_2 = {'catchment_name': 'Catchment 2', 'location': 'Region B'}
catchment_id_2 = 'C002'
catchment_2 = Catchment(dynamic_data_2, static_data_2, metadata_2, catchment_id_2)
```

### Step 2: Convert Catchments to a Dataset

```python
from hydroml.data.dataset import Dataset

catchments = [catchment_1, catchment_2]
dataset = Dataset.from_catchments(catchments)
```

# Load CAMELS-AUS Dataset

The [CAMELS-AUS](https://essd.copernicus.org/articles/13/3847/2021/) dataset contains both dynamic and static data for all the catchments in the CAMELS-AUS region. This dataset was the primary data source for training and experimenting with Hydroml. To load this dataset, we used the `Catchment` object method. The routine for this can be found in [hydroml.data.camels_aus_ds.py](../hydroml/data/camels_aus_ds.py). This routine is registered as a dataset in the `hydroml.data` module with the name `camels_aus_v1`. To load the dataset, you can use the following code:

```python
from hydroml.data import get_dataset
from hydroml.config import Config

config = Config(dataset_config={'dataset_config': {
            'name': 'camels_aus_v1',
            'static_feature_path_key': 'camels_aus_attributes',
            'dir_key': 'camels_aus_v1',
            'awra_dir_key': 'awra_postprocessed',
        }})

dataset = get_dataset(config, split_name='cal')
```

Here, 'name' refers to the CAMELS-AUS dataset. The 'static_feature_path_key', 'dir_key', and 'awra_dir_key' are configuration keys that map to paths containing:

static_feature_path_key: Path to the catchment attributes/static feature file
dir_key: Path to the main CAMELS-AUS dataset directory
awra_dir_key: Path to AWRA model outputs
These keys are mapped to actual filesystem paths in the path mapping configuration stored in the hydroml/config/path_mapping/ directory.

The split_name is used to specify which part of the data (both spatially and temporally) to load. The split definition is stored in the config. For more details, please refer to the [configuration](configuration.md) documentation.



# Reference
Fowler, K. J. A., Acharya, S. C., Addor, N., Chou, C., & Peel, M. C. (2021). CAMELS-AUS: hydrometeorological time series and landscape attributes for 222 catchments in Australia, Earth Syst. Sci. Data, 13, 3847–3867.

