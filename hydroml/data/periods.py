import pandas as pd
from hydroml.config.config import Config
from typing import List

def get_split(config: Config, split_name: str) -> List[List[str]]:
    VALID_SPLIT_NAMES = ['cal', 'val', 'test']
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(f"Split name {split_name} is not valid. Valid split names are {VALID_SPLIT_NAMES}")

    split = config.get(split_name)
    if split is None:
        raise ValueError(f"Split {split_name} is not defined in the config")
    
    return split


def make_periods(config: Config, split_name: str) -> pd.DatetimeIndex:
    # Create a list to store all dates

    periods = get_split(config, split_name)

    all_dates = []
    for start_date, end_date in periods['periods']:
        # Generate date range for each period
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates.extend(dates)
    
    # Combine all dates and remove duplicates
    return pd.DatetimeIndex(sorted(set(all_dates)))
