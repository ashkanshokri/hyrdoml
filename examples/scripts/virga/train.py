'''
temporal out of sample validation
'''

from hydroml.config.config import Config
from hydroml.workflow.evaluation import train_finetune_evaluate
import multiprocessing
from hydroml.utils import config_path_handler as cp


def get_config(name):
    with open(cp.get_package_base_dir() / 'examples' / 'sample_data' / 'basins.txt', 'r') as f:
        basins = f.read().splitlines()

    catchment_ids = [x for x in basins if x not in ['A5040517', '219001', '235205']] # no awra data for this catchment
    
    config = Config(cal={'periods' : [['1985-01-01', '2014-01-01']], 'catchment_ids':catchment_ids}, 
                    val={'periods' : [['1985-01-01', '2024-01-01']], 'catchment_ids':catchment_ids}, 
                    name=f'{name}_train_with_all_data_1985_2014',
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


    config = cp.update_config_paths(config, platform='virga')

    config.set_new_version_name()

    return config


def main(name):
    config = get_config(name)
    train_finetune_evaluate(config)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Use 'spawn' method
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()    
    main(name=args.name)
    # example: python validation.py toos_qc_validation

