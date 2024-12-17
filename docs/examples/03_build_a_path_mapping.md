# Set up the path mapping
 
The package can contain several different path mappings so for each platform we can use an appropriate path mapping file. The path mappings are stored in the config/path_mappings folder as JSON files. For this tutorial the path mapping needs to include 4 paths:
 
 - parent_path: The path to the parent directory where the model will be stored
 - camels_aus_attributes: The path to the CAMELS Australia attributes file
 - camels_aus_v1: The path to the CAMELS Australia v1 data
 - awra_postprocessed: The path to the AWRA postprocessed data
 
 In the next cell we are going to build a path mapping and save it as a JSON file in the config/path_mappings folder.



```python
from hydroml.utils import helpers as h
from hydroml.utils import config_path_handler as cp

path_mapping = {
    "parent_path": "//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/models/",
    "camels_aus_attributes": "//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/data/camels_aus/v1/CAMELS_AUS_Attributes&Indices_MasterTable.csv",
    "camels_aus_v1": "//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/data/camels_aus/v1//preprocessed", 
    "awra_postprocessed": "//fs1-cbr.nexus.csiro.au/{d61-coastal-forecasting-wp3}/work/sho108_handover/data/awra"
}

 
platform_name = 'win_2' # this can be any arbitrary name. So you can refer to the path mapping later.

path_mapping_file_path = cp.get_path_mapping_file(platform_name)

h.save_json(path_mapping, path_mapping_file_path)

```


```python

```
