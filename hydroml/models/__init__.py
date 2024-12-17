from .lstm import HydroLSTM
from hydroml import models 
import torch

def get_model(config):
    return getattr(models, config.model)


__all__ = ['HydroLSTM']
