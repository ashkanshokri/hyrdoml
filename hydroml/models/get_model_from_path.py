from hydroml.config.config import load_config
from pathlib import Path
from hydroml.models import get_model

def get_model_from_path(path, check_point='last'):
    config = load_config(Path(path)/'config.yaml')  
    Model = get_model(config)
    return Model.load_from_checkpoint(Path(path)/f'{check_point}.ckpt') #, config=config