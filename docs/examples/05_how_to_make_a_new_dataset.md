```python
pwd
```




    'p:\\work\\sho108\\hydroml\\examples\\notebooks'




```python
%load_ext autoreload
%autoreload 2

import hydroml.utils.helpers as h

all_folds = h.read_json('P:/work/sho108/hydroml/examples/4fold.json')
all_folds


from examples.scripts.virga.validation_xval import get_toos_cross_validation_dates

get_toos_cross_validation_dates(0, all_folds)



```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload





    ([['1970-01-01', '1975-01-01'], ['1990-01-01', '2015-01-01']],
     [['1975-01-01', '1985-01-01']])


