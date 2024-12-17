"""Functions for loading and processing CAMELS-AUS and AWRA datasets.

This module provides utilities to read catchment data from CSV files and create a Dataset object
for both the CAMELS-AUS and AWRA datasets. CAMELS-AUS provides observed streamflow and catchment 
attributes, while AWRA provides modeled streamflow data.
"""

import pandas as pd
from pathlib import Path
from typing import List

from tqdm import tqdm

from hydroml.config.config import Config
from hydroml.data.dataset import Dataset
from hydroml.data.catchment import Catchment
from hydroml.data.periods import get_split

def read_csv_camels(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return df

def read_csv_awra(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    return df


def get_dataset(config: Config, split_name: str = 'cal', is_train: bool = True) -> Dataset:
    """Create a Dataset from CAMELS-AUS catchment files.

    Args:
        data_dir: Directory containing the catchment CSV files.
        config: Configuration object with dataset parameters.

    Returns:
        Dataset object containing the loaded catchment data.
    """
    data_dir = Path(config.path_mapping[config.dataset_config['dir_key']])
    catchments: List[Catchment] = []

    split = get_split(config, split_name)

    
    catchment_ids = split['catchment_ids']
    with tqdm(total=len(catchment_ids), desc="Loading catchments", leave=False) as pbar:
        for i,catchment_id in enumerate(catchment_ids):
            pbar.set_description(f"Loading catchment {catchment_id}")
            
            df_camels = read_csv_camels(data_dir / f"{catchment_id}.csv")

            # only load awra data if it is specified in the config
            
            if 'awra_dir_key' in config.dataset_config:
                path = config.path_mapping[config.dataset_config['awra_dir_key']]
                df_awra = read_csv_awra(Path(path) / f"{catchment_id}.csv")
                df = pd.concat([df_camels, df_awra], axis=1)
            else:
                df = df_camels
            path = config.path_mapping[config.dataset_config['static_feature_path_key']]
            df_static  = pd.read_csv(path , index_col='station_id')
            
            static_features = df_static.loc[catchment_id, config.static_features].to_list()
            
            if 'weights' in split:
                weight = split['weights'][i]  
            else:
                weight = 1
            
            catchments.append(Catchment(df, static_features, id=catchment_id, metadata={'weight': weight}))
            pbar.update(1)

    dataset = Dataset.from_catchments(catchments, config, split_name, is_train)

    return dataset
