'''
temporal out of sample validation
'''

from hydroml.config.config import Config
from hydroml.workflow.evaluation import train_finetune_evaluate
import multiprocessing
import hydroml.utils.helpers as h

from hydroml.utils import config_path_handler as cp



def get_toos_cross_validation_dates(fold, all_folds, start_date='1970-01-01', end_date='2015-01-01', buffer_years=5):

    val_period = all_folds[str(fold)]
    cal_period = [[start_date, val_period[0]],
                   [h.add_years_to_date(val_period[1], buffer_years), end_date]]
    
    return cal_period, [val_period]
    

    
def get_config(name, fold):
    all_folds = h.read_json(cp.get_package_base_dir() / 'examples' / 'sample_data' / '4fold.json')
    basins_file = cp.get_package_base_dir() / 'examples' / 'sample_data' / 'basins.txt'

    with open(basins_file, 'r') as f:
        catchment_ids = f.read().splitlines()

    
    
    cal_period, val_period= get_toos_cross_validation_dates(fold, all_folds)
    config = Config(cal={'periods' : cal_period, 'catchment_ids':catchment_ids}, 
                    val={'periods' : val_period, 'catchment_ids':catchment_ids}, 
                    name = f'{name}_toos_xval_{fold}_of_{len(all_folds)}',
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


def main(name, fold):
    print(f'xval: running {name} for fold {fold}')
    
    config = get_config(name, fold)
    train_finetune_evaluate(config)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Use 'spawn' method

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('fold', type=int)
    args = parser.parse_args()
    


    main(name=args.name, fold=args.fold)

    # example: python validation.py toos_qc_validation

