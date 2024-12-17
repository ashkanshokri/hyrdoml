import numpy as np
import pandas as pd
from hydroml.data.catchment import Catchment
from hydroml.data.dataset import Dataset
from hydroml.config.config import Config

def make_catchment(id,start_date='2016-01-01', length=100, metadata=None):
    dynamic_features=['t_max_c', 't_min_c', 't_mean_c', 'precip_mm']
    target_features=['q_mm']
    features = dynamic_features + target_features
    data = {x: np.arange(length) for x in features}
    x_dynamic = pd.DataFrame(data, index = pd.date_range(start=start_date, periods=length, name='date')  )
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