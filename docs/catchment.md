# Catchment Object
The `Catchment` class is designed for managing hydrological catchment data, including dynamic, static, and metadata attributes. It provides methods to convert the data into an `xarray.Dataset`, facilitating multi-dimensional data analysis. Then a list of instance from this object can be uset to construct a dataset.


## CatchmentAttributes:
- **dynamic_data** (`pandas.DataFrame`): A DataFrame containing time-series data (e.g., streamflow, precipitation).
- **static_data** (`List[Any]`): A list representing static attributes (e.g., land area, vegetation type).
- **metadata** (`Dict[str, Any]`): A dictionary containing metadata about the catchment (e.g., name, location).
- **id** (`str`): A unique identifier for the catchment.

## Methods:

- **`__init__(self, dynamic_data: pd.DataFrame, static_data: List[Any], metadata: Dict[str, Any], id: str)`**  
  Initializes the `Catchment` object with dynamic data, static data, metadata, and a catchment ID.

- **`__getitem__(self, date: Union[str, pd.Timestamp])`**  
  A placeholder method for retrieving dynamic or static data for a specific date. This method is currently not implemented.

- **`metadata_to_dataarray(self)`**  
  Converts the metadata dictionary into an `xarray.DataArray`. This method returns the metadata values as an `xarray.DataArray` with the metadata keys as coordinates.

- **`to_xarray(self)`**  
  Converts the dynamic data, static data, and metadata into an `xarray.Dataset`. The resulting `Dataset` contains:
  - `dynamic`: Dynamic data as an `xarray.DataArray` with time and dynamic features.
  - `static`: Static data as an `xarray.DataArray`.
  - `metadata`: Metadata as an `xarray.DataArray`.
  - `catchment_id`: The catchment ID assigned as a coordinate.

## Catchment Example Usage

```python
import pandas as pd
from your_module import Catchment

# Example dynamic data (time-series)
dynamic_data = pd.DataFrame({
    'date': pd.date_range('2021-01-01', periods=5),
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [6, 7, 8, 9, 10]
})
dynamic_data.set_index('date', inplace=True)

# Static data and metadata
static_data = ['land_area', 'vegetation_type']
metadata = {'catchment_name': 'Catchment 1', 'location': 'Region A'}
catchment_id = 'C001'

# Initialize Catchment object
catchment = Catchment(dynamic_data, static_data, metadata, catchment_id)

# Convert to xarray Dataset
dataset = catchment.to_xarray()
print(dataset)
