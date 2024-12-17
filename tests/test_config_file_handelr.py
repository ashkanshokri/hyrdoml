import pytest
from pathlib import Path
from hydroml.utils import config_path_handler as cp

def test_get_path_mapping_file():
    # Test getting path mapping file for different platforms
    win_file = cp.get_path_mapping_file('win')
    assert win_file.name == 'win.json'
    assert win_file.exists()
    
    virga_file = cp.get_path_mapping_file('virga') 
    assert virga_file.name == 'virga.json'
    assert virga_file.exists()

def test_get_path_mapping_file_from_config():
    # Test with platform specified
    path_mapping = cp.get_path_mapping_file_from_config(platform='win')
    assert isinstance(path_mapping, dict)
    assert len(path_mapping) > 0

    # Test with invalid inputs
    with pytest.raises(ValueError):
        cp.get_path_mapping_file_from_config()
    
    with pytest.raises(ValueError):
        cp.get_path_mapping_file_from_config(
            path_mapping_file=Path('test.json'),
            platform='win'
        )

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        cp.get_path_mapping_file_from_config(
            path_mapping_file=Path('nonexistent.json')
        )

def test_update_config_paths():
    # Create mock config object
    class MockConfig:
        def __init__(self):
            self.path_mapping = {}
    
    config = MockConfig()
    
    # Test updating with platform
    updated_config = cp.update_config_paths(config, platform='win')
    assert isinstance(updated_config.path_mapping, dict)
    assert len(updated_config.path_mapping) > 0

    # Test updating with invalid inputs
    with pytest.raises(ValueError):
        cp.update_config_paths(config)

    with pytest.raises(ValueError):
        cp.update_config_paths(
            config,
            path_mapping_file=Path('test.json'),
            platform='win'
        )