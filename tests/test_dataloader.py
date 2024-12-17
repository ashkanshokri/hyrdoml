import pytest
import torch
from hydroml.config.config import Config
from hydroml.data import get_dataset
import numpy as np
import pandas as pd
from hydroml.data.catchment import Catchment
from hydroml.data.dataset import Dataset
import tempfile
 
@pytest.fixture
def basic_config():
    config = Config(
        device='cpu',
        target_features=['q_mm'],
        dynamic_features=['t_max_c', 't_min_c', 't_mean_c', 'precip_mm'],
        static_features=[],
        evolving_static_features={'precip_mm_mean': {'dynamic_feature': 'precip_mm', 'name': 'dynamic_feature_mean'},
                                   't_mean_c_mean': {'dynamic_feature': 't_mean_c', 'name': 'dynamic_feature_mean'}, },
        cal={'periods': [['2011-01-01', '2017-01-01']], 'catchment_ids':[1,2,3]},
        evolving_metadata={'observed_target_std': {'target_features': ['q_mm']}},
        batch_size=5,  # Reduced batch size since we only have 2 valid catchments
        seqlen=4,
        add_sin_cos_doy=False,
        dataloader_nworkers=0,
        dataloader_persistent_workers=False
    )
    config.set_new_version_name()
    temp_dir = tempfile.mkdtemp()
    config.parent_path = temp_dir
    
    return config

def make_catchment(id,start_date='2016-01-01', length=100, metadata=None):
    dynamic_features=['t_max_c', 't_min_c', 't_mean_c', 'precip_mm']
    target_features=['q_mm']
    features = dynamic_features + target_features
    data = {x: np.random.rand(length) for x in features}  # Use random data instead of sequential
    x_dynamic = pd.DataFrame(data, index = pd.date_range(start=start_date, periods=length, name='date'))
    # random missing values
    x_dynamic.iloc[np.random.choice(range(length), int(length*0.01), replace=False), 0] = np.nan
    x_dynamic.iloc[np.random.choice(range(length), int(length*0.1), replace=False), 1] = np.nan
    x_static = [0, 1.2]
    cat = Catchment(dynamic_data=x_dynamic, static_data=x_static, metadata=metadata, id=id)
    return cat

def make_dataset(config: Config, split_name: str = 'cal'):
    cat1 = make_catchment(1, start_date='2000-01-01', length=2000, metadata={'md': 0})
    cat2 = make_catchment(2, start_date='2006-01-01', length=2000, metadata={'md1': 1})
    cat3 = make_catchment(3, start_date='2016-01-01', length=1000, metadata={'md': 2})
    ds = Dataset.from_catchments([cat1, cat2, cat3], config, split_name)
    return ds

def test_dataloader_batch_contents(basic_config):
    """Test if dataloader produces batches with expected contents and shapes."""
    dataset = make_dataset(basic_config, split_name='cal')
    print(f'dataset.x_dynamic.shape = {dataset.x_dynamic.shape}')
    dataloader = dataset.to_dataloader()
    
    batch = next(iter(dataloader))
    
    print(f'batch = {batch.keys()}')
    # Test batch contains expected keys
    assert set(batch.keys()) >= {'x_static', 'x_dynamic', 'y', 'metadata'}
    
    # Test x contains dynamic and static features
    assert isinstance(batch['x_static'], torch.Tensor)
    assert isinstance(batch['x_dynamic'], torch.Tensor)
    assert isinstance(batch['y'], torch.Tensor)
    assert isinstance(batch['metadata'], dict)
    
    # Test shapes
    batch_size = basic_config.batch_size
    seq_len = dataset.seqlen
    n_dynamic = len(basic_config.dynamic_features) + (2 if basic_config.add_sin_cos_doy else 0)
    n_static = len(basic_config.static_features) + len(basic_config.evolving_static_features)

    assert batch['x_dynamic'].shape == (batch_size, seq_len, n_dynamic)
    print(f'batch["x_static"].shape = {batch["x_static"].shape}, {(batch_size, n_static)}')
    assert batch['x_static'].shape == (batch_size, n_static)
    
    print(f'batch["y"].shape = {batch["y"].shape}, {(batch_size, basic_config.number_of_time_output_timestep, len(basic_config.target_features))}')
    
    
    # Test all tensors are on correct device
    assert batch['x_dynamic'].device.type == basic_config.device
    assert batch['x_static'].device.type == basic_config.device
    assert batch['y'].device.type == basic_config.device
