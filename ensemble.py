import os
import pandas as pd
from config import cfg

### Bagging

# Travel Time
args = ['']
for index, name in args:
	file_name = os.path.join(cfg.data.prediction_dir, name)
    if index == 0:
        result = pd.read_csv(file_name)
    else:
        df = pd.read_csv(file_name)
        result['avg_travel_time'] = result['avg_travel_time'] + df['avg_travel_time']
    result['avg_travel_time'] = result['avg_travel_time'] / len(args)
result.to_csv('./data/prediction/{}_ensemble_travelTime.csv'.format(exp_name), sep=',', header=True,index=False)
