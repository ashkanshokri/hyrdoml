from hydroml.data import camels_aus_ds

def get_dataset(config, split_name, is_train=True):
    if config.dataset_config['name'] == 'camels_aus_v1':
        return camels_aus_ds.get_dataset(config, split_name, is_train)
    else:
        raise ValueError(f"Dataset {config.dataset_config['name']} not found")