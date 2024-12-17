'''
temporal out of sample validation
'''

from hydroml.config.config import Config
from hydroml.workflow.evaluation import train_finetune_evaluate
import multiprocessing

from hydroml.utils import config_path_handler as cp


def get_config(name):
    basins_file = cp.get_package_base_dir() / 'examples' / 'sample_data' / 'basins.txt'
    with open(basins_file, 'r') as f:
        catchment_ids = f.read().splitlines()

    
    config = Config(cal={'periods' : [['1991-01-01', '2014-01-01']], 'catchment_ids':catchment_ids}, 
                    val={'periods' : [['1985-01-01', '1990-01-01']], 'catchment_ids':catchment_ids}, 
                    name = name,
                    device='cuda',
                    platform='virga'
                    )    
    
    if name == 'toos_qc_validation':
        config.dynamic_features = ['precipitation_AWAP', 'et_morton_wet_SILO', 'qtot']
    
    elif name == 'toos_c_validation':
        config.dynamic_features = ['precipitation_AWAP', 'et_morton_wet_SILO']
    
    elif name == 'toos_qc_validation_silo_silo':
        config.dynamic_features = ['precipitation_SILO', 'et_morton_wet_SILO', 'qtot']
    
    elif name == 'toos_c_validation_silo_silo':
        config.dynamic_features = ['precipitation_SILO', 'et_morton_wet_SILO']
    
    elif name == 'toos_q_validation':
        config.dynamic_features = ['qtot']    

    

    config.set_new_version_name()

    return config


def main(name):
    
    config = get_config(name)
    train_finetune_evaluate(config)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    
    main(name=args.name)


