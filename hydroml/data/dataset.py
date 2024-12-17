#  imports
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

# hydroml
import hydroml.utils.helpers as h
from hydroml.config.config import Config

# hydroml data
from hydroml.data.evolving_static_features import get_evolving_dynamic_feature_function # type: ignore
from hydroml.data.periods import get_split, make_periods # type: ignore
from hydroml.data.transformers import get_transformer # type: ignore


def observed_target_std(dataset: xr.Dataset, kwargs: dict):
    return dataset['dynamic'].sel(dynamic_feature=kwargs['target_features']).std(dim='date').values

def get_evolving_metadata_function(feature: str):
    if feature == 'observed_target_std':        
        return observed_target_std


class Dataset(TorchDataset):
    NAME = 'Dataset'
    
    def __init__(self, dataset: xr.Dataset,
                 config: Config,
                 split_name: str, 
                 is_train: bool = True):
        """Initialize a Dataset object.

        Args:
            dataset (xr.Dataset): Dataset containing catchment data with dynamic and static features.
            config (Config): Configuration object containing dataset parameters.
            split_name (str): Name of the data split ('cal', 'val', or 'test').
            is_train (bool, optional): Whether this is a training dataset. Defaults to True.
        """

        self.seqlen = config.seqlen
        self.config = config
        self.split_name = split_name    
        self.is_train = is_train
        
        self.dataset = dataset
        self.metadata = {}
        self.initialize_dataset()

    @classmethod
    def from_catchments(cls, catchments: List[str], *args, **kwargs):
        dataset = xr.concat([catchment.to_xarray() for catchment in catchments], dim='catchment_id')
        return cls(dataset, *args, **kwargs)
    

    def valid_data_points_per_catchment(self):
        pass

    def initialize_dataset(self):
        # subset the dataset according to the split both in time and catchments
        

        self.apply_split() # this can be somewhete eles. this is the only place that we are implementing change to the input dataset and subsetting it.

        self.dynamic_features = self.config.dynamic_features
        self.target_features = self.config.target_features
        self.static_features = self.config.static_features

        
        missing_features = [feature for feature in self.target_features if feature not in self.dataset['dynamic'].coords['dynamic_feature'].values]
        
        new_dynamic_features = np.concatenate([self.dataset['dynamic'].coords['dynamic_feature'].values, missing_features])
        #self.dataset['dynamic'] = self.dataset['dynamic'].reindex({"dynamic_feature": new_dynamic_features}, fill_value=np.nan)

        self.dataset = self.dataset.reindex({"dynamic_feature": new_dynamic_features}, fill_value=np.nan)
        
        

        try:
            self.x_dynamic : np.ndarray = self.dataset['dynamic'].sel(dynamic_feature=self.dynamic_features).values
        except KeyError:
            raise ValueError(f"Dynamic features {self.dynamic_features} are not in the dataset")

        try:
            self.y : np.ndarray = self.dataset['dynamic'].sel(dynamic_feature=self.target_features).values
        except KeyError:

            raise ValueError(f"Target features {self.target_features} are not in the dataset")

        if self.static_features:    
            try:
                self.x_static : np.ndarray = self.dataset['static'].values
            except KeyError:
                raise ValueError(f"Static features {self.static_features} are not in the dataset")
        else:
            self.x_static = np.empty((self.dataset.sizes['catchment_id'], 0))

        # Adding additional features to the dataset.
        # Calculate the sine and cosine values for the day of the year using the 'date' variable from the dataset.
        # The resulting tensor, sin_cos_doy, has a shape of (number_of_dates, 2), where the first column contains 
        # the sine values and the second column contains the cosine values.
        if self.config.add_sin_cos_doy:
            self.sin_cos_doy = h.get_sin_cos_doy(self.dataset['date'])
        else:
            self.sin_cos_doy = None

        # read the metadata
        self.metadata = {
            dim.values.tolist(): self.dataset.metadata.sel(metadata_dim=dim).values
            for dim in self.dataset.metadata_dim
        }

        # transform: implement the transforms on numpy arrays ie x_dynamic and y    
        self.transform()

        self.add_evolving_static_features()# TODO: add other evolving static features, such as aridity index, 

        self.add_evolving_metadata()
        
        # make a list of coordinates (catchment_id, idx) of valid data points. idx is the day of target in which the sequence before that does not have nans        
        self.indexs = self.make_index()

        self.convert_to_torch()



    def apply_split(self):
        """
        Subset the dataset according to the split both in time and catchments.
        """

        # apply the split in time
        self.periods = make_periods(self.config, self.split_name)
        #overlap the dataset with the periods
        overlapped_dates = self.periods.intersection(self.dataset['date'].values)
        self.dataset = self.dataset.sel(date=overlapped_dates)
        self.dataset = self.dataset.sortby('date')
        # reindex by date to make sure all dates are available
        self.dataset = self.dataset.reindex(date=pd.date_range(start=self.dataset['date'].values[0], end=self.dataset['date'].values[-1], freq='D'))

        # apply the split in catchments
        self.catchments = get_split(self.config, self.split_name)['catchment_ids']
        
        # This raises an error if the catchments are not in the dataset. Intentionally left if unhandled to make sure the catchments are correct.
        self.dataset = self.dataset.sel(catchment_id=self.catchments)

        # Find catchments where all dynamic data is NaN
        if self.config.drop_catchments_with_all_nans: 
            all_nan_catchments = self.dataset.dynamic.isnull().all(dim=["date", "dynamic_feature"])
            if all_nan_catchments.any():    
                print(f'dropping catchments due to all nans: {self.dataset.catchment_id[all_nan_catchments].values}')
                # Drop these catchments
                self.dataset = self.dataset.drop_sel(catchment_id=self.dataset.catchment_id[all_nan_catchments])
                
        
    def add_evolving_static_features(self):
        for name, kwargs in self.config.evolving_static_features.items():
            evolving_static_feature_function = get_evolving_dynamic_feature_function(kwargs['name'])
            new_x_static = evolving_static_feature_function(self.dataset, kwargs['dynamic_feature']) # TODO: this should be per catchment


            if len(new_x_static.shape) == 1:
                new_x_static = new_x_static.reshape(-1, 1)
            
            # merge on axis 1 which is static feature dimension in x_static (axis 0 is catchment dimension)
            self.x_static = np.concatenate([self.x_static, new_x_static], axis=1)

    def add_evolving_metadata(self):
        for k, kwargs in self.config.evolving_metadata.items():
            evolving_metadata_function = get_evolving_metadata_function(k)
            new_metadata = evolving_metadata_function(self.dataset, kwargs) # TODO: this should be per catchment
            
            # merge on axis 1 which is static feature dimension in x_static (axis 0 is catchment dimension)
            self.metadata[k] = new_metadata

    def transform(self):
        transform_parameter_path = self.config.get_transform_parameters_path()
        
        # calculate the parameters
        if self.is_train and not transform_parameter_path.exists():
            print(f'Transforming data: calculating transform parameters and saving to {transform_parameter_path}')
            # calculate the parameters
            parameters = {}
            for i, feature in enumerate(self.dynamic_features):
                transformer_dynamic = get_transformer(self.config.transform_predictor_dynamic)()
                parameters[feature] = transformer_dynamic.calculate_parameters(self.x_dynamic[:, :, i])  
            
            for i, feature in enumerate(self.target_features):
                transformer_target = get_transformer(self.config.transform_target)()
                parameters[feature] = transformer_target.calculate_parameters(self.y[:, :, i])
                
            h.save_yaml(parameters, transform_parameter_path)

        else:
            print(f'Transforming data: loading transform parameters from {transform_parameter_path}')
            parameters = h.read_yaml(transform_parameter_path)



        # transform the data
        for i, feature in enumerate(self.dynamic_features):
            transformer_dynamic = get_transformer(self.config.transform_predictor_dynamic)(parameters[feature])
            self.x_dynamic[:, :, i] = transformer_dynamic.transform(self.x_dynamic[:, :, i])

        for i, feature in enumerate(self.target_features):
            transformer_target = get_transformer(self.config.transform_target)(parameters[feature])
            self.y[:, :, i] = transformer_target.transform(self.y[:, :, i])


    def convert_to_torch(self):
        self.x_dynamic = torch.from_numpy(self.x_dynamic.astype(np.float32)).to(self.config.device)
        self.x_static = torch.from_numpy(self.x_static.astype(np.float32)).to(self.config.device)
        self.y = torch.from_numpy(self.y.astype(np.float32)).to(self.config.device)
        if self.sin_cos_doy is not None:
            self.sin_cos_doy = torch.from_numpy(self.sin_cos_doy.astype(np.float32)).to(self.config.device)
        self.metadata = {k: torch.from_numpy(np.array(v).astype(np.float32)).to(self.config.device) for k, v in self.metadata.items()}
    

    def __getitem__(self, index: int) -> Tuple[xr.Dataset, xr.Dataset]:
        catchment_id, idx = self.indexs[index]
        start_idx, end_idx = self.get_idx_range(idx)
        #x_dynamic = torch.from_numpy(self.x_dynamic[catchment_id, start_idx:end_idx, :].astype(np.float32))
        x_dynamic = self.x_dynamic[catchment_id, start_idx:end_idx, :]
        x_static = self.x_static[catchment_id, :]
        
        metadata = {k: v[catchment_id] for k, v in self.metadata.items()}



        y = self.y[catchment_id, idx:idx+1, :] # TODO: should it be idx+1?

        if self.sin_cos_doy is not None:
            sin_cos_doy = self.sin_cos_doy[start_idx:end_idx, :]
            x_dynamic = torch.cat([x_dynamic, sin_cos_doy], dim=1)

        # return the target date. It might need to be converted to Torch later.
        date = self.dataset['date'].isel(date=idx).values.tolist() # to convert it to int, use pd.to_datetime to convert it back to datetime
        catchment_id = self.dataset['catchment_id'].isel(catchment_id=catchment_id).values.tolist()
        
        return {'x_dynamic': x_dynamic, 'x_static': x_static, 'y': y, 'date': date, 'catchment_id': catchment_id, 'metadata': metadata}

    def __len__(self) -> int:
        return len(self.indexs)

    def get_idx_range(self, idx: int) -> Tuple[int, int]:
        start_idx = idx - self.seqlen + 1

        # Add 1 to include the current day (idx) in the sequence range
        end_idx = idx + 1
        return start_idx, end_idx

    def make_index(self):
        valid_coordinates = []
        self.valid_data_points_per_catchment = {}
        for catchment_id in range(self.x_dynamic.shape[0]):
            for idx in range(self.seqlen, self.x_dynamic.shape[1]):
                start_idx, end_idx = self.get_idx_range(idx)
                if all(~np.isnan(self.x_dynamic[catchment_id, start_idx:end_idx, :]).flatten()):
                    # if we are in training mode, we need to check if the target is not nan,
                    if self.is_train:
                        if all(~np.isnan(self.y[catchment_id, start_idx:end_idx, :]).flatten()):
                            valid_coordinates.append((catchment_id, idx))
                            self.valid_data_points_per_catchment[catchment_id] = self.valid_data_points_per_catchment.get(catchment_id, 0) + 1
                    #  for validation we do not need to check the target, 
                    # and only if the x_dynamic is not nan we can make the prediction
                    else:
                        valid_coordinates.append((catchment_id, idx))
                        self.valid_data_points_per_catchment[catchment_id] = self.valid_data_points_per_catchment.get(catchment_id, 0) + 1
        return valid_coordinates        
        
    def to_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Converts the dataset into a PyTorch DataLoader.

        Args:
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            DataLoader: A PyTorch DataLoader configured with the dataset's settings.
        """
        return DataLoader(
            self,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_nworkers, 
            persistent_workers=self.config.dataloader_persistent_workers,
            shuffle=shuffle
        )
    
    # show the dataset when ds is called in a cell
    def _repr_html_(self) -> str:
        return self.dataset._repr_html_().replace('xarray.Dataset', self.NAME)

    def __repr__(self) -> str:
        return self.dataset.__repr__()

    def __str__(self) -> str:
        return self.dataset.__str__()



