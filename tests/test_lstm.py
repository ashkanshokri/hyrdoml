import pytest
import torch
from hydroml.config.config import Config
from hydroml.models.lstm import HydroLSTM


@pytest.fixture
def basic_config():
    return Config(
        lstm_dynamic_input_feature_size=10,
        lstm_static_input_size=1,
        lstm_dynamic_input_feature_latent_size=5,
        lstm_static_input_latent_size=1,
        lstm_hidden_size=32,
        lstm_layers=2,
        dropout_probability=0.1,
        number_of_time_output_timestep=1
    )


@pytest.fixture
def sample_batch(basic_config):
    batch_size = 16
    seq_len = 10
    
    return {
        'x_dynamic': torch.randn(batch_size, seq_len, basic_config.lstm_dynamic_input_feature_size),
        'x_static': torch.randn(batch_size, basic_config.lstm_static_input_size),
        'y': torch.randn(batch_size, basic_config.number_of_time_output_timestep),
        'metadata': {
            'observed_target_std': torch.randn(batch_size, 1),
            'weight': torch.randn(batch_size, 1)
        }
    }


def test_model_initialization(basic_config):
    """Test if model initializes correctly with given config."""
    model = HydroLSTM(basic_config)
    assert isinstance(model, HydroLSTM)
    assert model.lstm_dynamic_input_feature_size == basic_config.lstm_dynamic_input_feature_size
    assert model.lstm_static_input_size == basic_config.lstm_static_input_size


def test_forward_pass_shape(basic_config, sample_batch):
    """Test if forward pass returns correct output shape."""
    model = HydroLSTM(basic_config)
    output = model(sample_batch['x_dynamic'], sample_batch['x_static'])
    
    expected_shape = (
        sample_batch['x_dynamic'].shape[0],  # batch size
        basic_config.number_of_time_output_timestep,  # output timesteps
        basic_config.lstm_target_size  # target size (usually 1)
    )
    
    assert output.shape == expected_shape


def test_loss_computation(basic_config, sample_batch):
    """Test if loss computation works."""
    model = HydroLSTM(basic_config)
    loss = model.loss(sample_batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss
    assert not torch.isnan(loss)


def test_freeze_unfreeze_layers(basic_config):
    """Test layer freezing and unfreezing functionality."""
    model = HydroLSTM(basic_config)
    
    # Test freezing
    model.freeze_all_layers()
    for param in model.parameters():
        assert not param.requires_grad
    
    # Test unfreezing
    model.unfreeze_all_layers()
    for param in model.parameters():
        assert param.requires_grad


def test_merge_static_dynamic():
    """Test static and dynamic feature merging."""
    batch_size = 4
    seq_len = 5
    dynamic_size = 3
    static_size = 2
    
    dynamic = torch.randn(batch_size, seq_len, dynamic_size)
    static = torch.randn(batch_size, static_size)
    
    merged = HydroLSTM.merge_static_dynamic(dynamic, static)
    
    assert merged.shape == (batch_size, seq_len, dynamic_size + static_size) 