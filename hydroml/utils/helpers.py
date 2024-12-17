# imports
import yaml
import json
from typing import Any
import datetime
import torch
import numpy as np
import xarray as xr
import uuid
from pathlib import Path


def read_json(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)
    
    
    
def save_json(data: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def read_yaml(path: str) -> dict[str, Any]:
    """Reads a YAML file and returns its contents as a dictionary.

    Args:
        path (str): The path to the YAML file.

    Returns:
        dict[str, Any]: The contents of the YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def save_yaml(data: dict[str, Any], path: str) -> None:
    """Saves a dictionary to a YAML file.

    Args:
        data (dict[str, Any]): The data to save.
        path (str): The path where the YAML file will be saved.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def now() -> str:
    """Returns the current date and time as a formatted string.

    Returns:
        str: The current date and time in 'yymmddHHMMSS' format.
    """
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')

def get_version_name():
    return now() + '_' + str(uuid.uuid4())[:4]


def get_sin_cos_doy(date: xr.DataArray) -> torch.Tensor:
    """Calculate the sine and cosine of the day of the year for a given date.

    Args:
        date (pd.DatetimeIndex): A pandas DatetimeIndex object.

    Returns:
        torch.Tensor: A tensor containing the sine and cosine values of the day of the year.
    """
    doy = date.dt.dayofyear
    # Normalize the day of the year to a range of [0, 1] for the sine and cosine functions
    normalized_doy = doy / 365.0
    sin_doy = np.sin(2 * np.pi * normalized_doy)
    cos_doy = np.cos(2 * np.pi * normalized_doy)
    return np.column_stack((sin_doy, cos_doy))


def parse_kwargs(kwargs_list):
    kwargs = {}
    for arg in kwargs_list:
        key, value = arg.split('=')
        print(value, end='-')
        
        # Check if the value is a list
        if value == '[]':
            value = []
        elif value.startswith('[') and value.endswith(']'):
            # Convert the string representation of the list into a Python list
            value = value[1:-1].split(';')
        else:
            value = auto_number_convert(value)

        kwargs[key] = value
        

    
    return kwargs

def auto_number_convert(value):
    """
    Try to convert the value to an int or float, if possible.
    """
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
    

def add_years_to_date(date_str, years):
    """
    Add a specified number of years to a date string in 'YYYY-MM-DD' format.

    Args:
    - date_str (str): The date string to add years to, in 'YYYY-MM-DD' format.
    - years (int): The number of years to add.

    Returns:
    - str: The new date string after adding the specified number of years.
    """
    # Convert the string to a datetime object
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')

    try:
        # Add the specified number of years
        new_date_obj = date_obj.replace(year=date_obj.year + years)
    except ValueError:
        # Handles cases like adding to February 29 on a non-leap year
        new_date_obj = date_obj.replace(year=date_obj.year + years, day=date_obj.day - 1)
    
    # Convert the datetime object back to a string
    return new_date_obj.strftime('%Y-%m-%d')