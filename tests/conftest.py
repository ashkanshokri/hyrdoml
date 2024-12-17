import pytest
import torch
import numpy as np

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    return None 