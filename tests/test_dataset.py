import pytest
import torch
import xarray as xr
from hydroml.config.config import Config
from hydroml.data import get_dataset
import pandas as pd
import numpy as np
import tempfile
 
@pytest.fixture
def basic_config():
    config = Config(device='cpu')
    temp_dir = tempfile.mkdtemp()
    config.parent_path = temp_dir
    
    config.set_new_version_name()
    return config


@pytest.fixture
def sample_dataset():
    # Create a minimal xarray dataset for testing
    dates = pd.date_range('2000-01-01', '2000-12-31')
    stations = ['station1', 'station2']
    
    dynamic_data = xr.DataArray(
        np.random.randn(len(stations), len(dates), 3),
        dims=['station', 'date', 'dynamic_feature'],
        coords={
            'station': stations,
            'date': dates,
            'dynamic_feature': ['feature1', 'feature2', 'feature3']
        }
    )
    
    static_data = xr.DataArray(
        np.random.randn(len(stations), 2),
        dims=['station', 'static_feature'],
        coords={
            'station': stations,
            'static_feature': ['static1', 'static2']
        }
    )
    
    return xr.Dataset({
        'dynamic': dynamic_data,
        'static': static_data
    })


def test_get_dataset(basic_config):
    """Test if dataset can be retrieved successfully."""
    dataset = get_dataset(basic_config, split_name='cal')
    assert dataset is not None


def test_dataset_attributes(basic_config, sample_dataset):
    """Test if dataset has expected attributes."""
    dataset = get_dataset(basic_config, split_name='cal')
    assert hasattr(dataset, 'seqlen')
    assert hasattr(dataset, 'config')
    assert hasattr(dataset, 'split_name')
    assert hasattr(dataset, 'is_train')
    assert hasattr(dataset, 'dataset')
    assert hasattr(dataset, 'metadata')
