import os
import pandas as pd
from config import cfg

### Bagging

# Travel Time
args = ['RNN-0', 'RNN-1', 'RNN-2', 'RNN-3']
for index, name in enumerate(args):
	file_name = os.path.join(cfg.data.prediction_dir, name + '_travelTime.csv')
	if index == 0:
	    result_travel = pd.read_csv(file_name)
	else:
	    df = pd.read_csv(file_name)
	    result_travel['avg_travel_time'] = result_travel['avg_travel_time'] + df['avg_travel_time']
result_travel['avg_travel_time'] = result_travel['avg_travel_time'] / len(args) * 10
result_travel.to_csv('./data/prediction/{}_ensemble_travelTime.csv'.format('5_16'), sep=',', header=True,index=False)
