from pathlib import Path
from hydroml.utils import helpers as h

from typing import Dict
def get_package_base_dir() -> Path:
    return Path(__file__).parent.parent.parent

def get_path_mapping_file(platform) -> Path:
    return get_package_base_dir() / 'hydroml' / 'config' / 'path_mappings' / f'{platform}.json'


def get_path_mapping_file_from_config(platform=None, path_mapping_file=None) -> Dict[str, str]:
    """Load path mappings from a JSON configuration file.
    
    Args:
        path_mapping_file: Optional path to a JSON file containing custom path mappings. 
            If provided, platform must be None.
        platform: Optional platform name (e.g. 'win', 'linux') to load the default mapping file for that platform.
            If provided, path_mapping_file must be None.
            
    Returns:
        dict: Dictionary containing the path mappings loaded from the JSON file
        
    Raises:
        ValueError: If neither or both path_mapping_file and platform are provided
        FileNotFoundError: If the specified path mapping file does not exist
    """
    if path_mapping_file is not None and platform is not None:
        raise ValueError("Only one of path_mapping_file or platform should be provided")
        
    if path_mapping_file is None and platform is None:
        raise ValueError("Either path_mapping_file or platform must be provided")
        
    if platform is not None:
        path_mapping_file = get_path_mapping_file(platform)
        
    if not Path(path_mapping_file).exists():
        raise FileNotFoundError(f"Path config file not found: {path_mapping_file}")
    
    return h.read_json(path_mapping_file)
    


def update_config_paths(config, path_mapping_file=None, platform=None):
    """Update the path mappings in a config object using a JSON configuration file.
    
    This function loads path mappings from a JSON file and updates the config.path_mapping attribute.
    The path mappings define the locations of data and output directories for different platforms.
    
    Args:
        config: The config object to update
        path_mapping_file: Optional path to a JSON file containing custom path mappings.
            If provided, platform must be None.
        platform: Optional platform name (e.g. 'win', 'linux') to load the default mapping file for that platform.
            If provided, path_mapping_file must be None.
        
    Returns:
        Config: The config object with updated path_mapping attribute
    """
    config.path_mapping = get_path_mapping_file_from_config(path_mapping_file, platform)

    return config
