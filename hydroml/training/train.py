
from hydroml.utils import helpers as h
from hydroml.data import get_dataset
from hydroml.training.trainer import get_trainer
from hydroml.models import get_model

def train(config, cal_split_name: str = 'cal', val_split_name: str = 'val'):

    if config.version == 'none' or config.version == None:
        config.version = h.get_version_name()
    
    config.make_dirpath()
    # build dataloaders
    cal_dataset = get_dataset(config, cal_split_name)
    cal_dataloader = cal_dataset.to_dataloader(shuffle=True)
    print('valid data points per catchment',   cal_dataset.valid_data_points_per_catchment)

    val_dataset = get_dataset(config, val_split_name)
    val_dataloader = val_dataset.to_dataloader()

    # save config
    config.save_to_yaml()

    # make model
    Model = get_model(config)
    model = Model(config)
    

    # make trainer
    
    trainer = get_trainer(config)

    trainer.fit(model, cal_dataloader, val_dataloader)

    return config.current_path, config.version


