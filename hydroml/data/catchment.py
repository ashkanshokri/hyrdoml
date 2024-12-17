import pandas as pd
from typing import Dict, Any, Union, List
import xarray as xr
class Catchment():
    def __init__(self, dynamic_data: pd.DataFrame, static_data: List[Any], metadata: Dict[str, Any], id: str):
        self.dynamic_data = dynamic_data
        self.static_data = static_data
        self.metadata = metadata
        self.id = id

    def __getitem__(self, date: Union[str, pd.Timestamp]) -> Dict[str, Any]:
        NotImplementedError

    def metadata_to_dataarray(self):
        metadata_da = xr.DataArray(list(self.metadata.values()), dims=['metadata_dim'], coords={'metadata_dim': list(self.metadata.keys())})
        return metadata_da

    def to_xarray(self):
        #ds = xr.Dataset.from_dataframe(self.dynamic_data)
        dynamic_da = xr.DataArray(self.dynamic_data.values, coords={'date': self.dynamic_data.index.to_list(), 'dynamic_feature': self.dynamic_data.columns.to_list()})
        static_da = xr.DataArray(self.static_data, dims=['static_feature_dim'])
        metadata_da = self.metadata_to_dataarray()
        ds = xr.Dataset({'dynamic': dynamic_da, 'static': static_da, 'metadata': metadata_da})
        ds.attrs.update(self.metadata)
        ds = ds.assign_coords({'catchment_id': self.id})
        return ds
    

    