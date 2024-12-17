import os
import shutil
from pathlib import Path
from hydroml.utils import config_path_handler as cp
from tqdm import tqdm


tutorials_folder = cp.get_package_base_dir() / 'examples' / 'notebooks/'
documentation_folder = cp.get_package_base_dir() / 'docs' / 'examples'



# remove documentation folder if exists
if documentation_folder.exists():
    shutil.rmtree(documentation_folder)

documentation_folder.mkdir(parents=True, exist_ok=True) 

for notebook in tqdm(tutorials_folder.glob('*.ipynb')):
    
    os.system(f'jupyter nbconvert --to markdown {notebook}')


    name = notebook.stem
    
    shutil.move(notebook.with_suffix('.md'), documentation_folder / f'{name}.md')
    
    # move files folder as well for example 01_simulation_files
    if (notebook.parent / f'{name}_files').exists():
        shutil.move(notebook.parent / f'{name}_files', documentation_folder / f'{name}_files')
     
