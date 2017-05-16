from datetime import datetime, timedelta
from easydict import EasyDict as edict
import numpy as np

cfg = edict()
cfg.data = edict()
cfg.time = edict()
cfg.model = edict()

#### data directory
cfg.data.rawdata_dir = 'C:/Users/user/PycharmProjects/KDDCup 2017/data/dataSets/'
cfg.data.feature_dir = 'C:/Users/user/PycharmProjects/KDDCup 2017/data/features/'
cfg.data.checkpoint_dir = 'C:/Users/user/PycharmProjects/KDDCup 2017/data/checkpoint/'
cfg.data.prediction_dir = 'C:/Users/user/PycharmProjects/KDDCup 2017/data/prediction/'
cfg.data.validation_ratio = 0.03

#### time
cfg.time.time_interval = 20

# trajectory time range
cfg.time.trajectory_train_start = datetime.strptime("2016-07-19 00:00:00", "%Y-%m-%d %H:%M:%S")
cfg.time.trajectory_train_end   = datetime.strptime("2016-10-18 00:00:00", "%Y-%m-%d %H:%M:%S")
cfg.time.trajectory_totalmin = (cfg.time.trajectory_train_end - cfg.time.trajectory_train_start).total_seconds() / 60
cfg.time.trajectory_slots = cfg.time.trajectory_totalmin / cfg.time.time_interval


# volume time range
cfg.time.volume_train_start = datetime.strptime("2016-09-19 00:00:00", "%Y-%m-%d %H:%M:%S")
cfg.time.volume_train_end   = datetime.strptime("2016-10-18 00:00:00", "%Y-%m-%d %H:%M:%S")
cfg.time.volume_totalmin = (cfg.time.volume_train_end - cfg.time.volume_train_start).total_seconds() / 60
cfg.time.volume_slots = cfg.time.volume_totalmin / cfg.time.time_interval

cfg.time.festival = ["2016-09-{}".format(item) for item in range(15, 18)] + ["2016-10-{}".format(item) for item in range(1, 8)]
# testing timeslots
cfg.time.test_timeslots = []

for i in range(18, 25):
    for k in ['06', '15']:
        left = datetime.strptime("2016-10-{} {}:00:00".format(i, k), "%Y-%m-%d %H:%M:%S")
        for j in range(int(4*60/cfg.time.time_interval)):
            cfg.time.test_timeslots.append(left.strftime("%Y-%m-%d %H:%M:%S"))
            left = left + timedelta(minutes=cfg.time.time_interval)

# training timeslots
cfg.time.train_timeslots = []
right = cfg.time.trajectory_train_start
for slot in range(int(cfg.time.trajectory_slots)):
    left = right
    right = left + timedelta(minutes=cfg.time.time_interval)
    cfg.time.train_timeslots.append(left.strftime("%Y-%m-%d %H:%M:%S"))

# all_timeslots  = train timeslot + test timeslot
cfg.time.all_timeslots = cfg.time.train_timeslots.copy()
cfg.time.all_timeslots.extend(cfg.time.test_timeslots)

#split dataset into training set and validation set
validation_start = len(cfg.time.train_timeslots) - int(cfg.data.validation_ratio * len(cfg.time.train_timeslots))
cfg.time.validation_timeslots = cfg.time.train_timeslots[validation_start:]
cfg.time.train_timeslots = cfg.time.train_timeslots[:validation_start]
# cfg.time.train_timeslots = cfg.time.train_timeslots[3000:]

cfg.time.validation_timeslots.sort()
cfg.time.train_timeslots.sort()
cfg.time.test_timeslots.sort()
cfg.time.all_timeslots.sort()

print ("num of 20-minute window in training set: {}".format(len(cfg.time.train_timeslots)))
print ("num of 20-minute window in validation set: {}".format(len(cfg.time.validation_timeslots)))
print ("num of 20-minute window in testing set: {}".format(len(cfg.time.test_timeslots)))

assert len(np.unique(np.array(cfg.time.all_timeslots))) == len(cfg.time.all_timeslots), 'Time slots not unique'

#### Model

# loss scale
cfg.model.loss_scale = []
coeff = 1.0

for time in cfg.time.train_timeslots:
    tmp = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    coeff *= 1.00002
    if (tmp.hour>=8 and tmp.hour<=10) or (tmp.hour>=17 and tmp.hour<=19):
        cfg.model.loss_scale.append(1*coeff)
    else:
        cfg.model.loss_scale.append(1*coeff)
cfg.model.loss_scale.extend([0] * 12)
cfg.model.loss_scale = np.array(cfg.model.loss_scale)

# RNN topology
cfg.model.link = {# links
            '100': ['105','111'],
            '101': ['116','121'],
            '102': ['115','109'],
            '103': ['111','122','116'],
            '104': ['109','112'],
            '105': ['B'  ,'100'],
            '106': ['121','113'],
            '107': ['123','108'],
            '108': ['107','119','120'],
            '109': ['102','104'],
            '110': ['A', '123'],
            '111': ['100', '112', '103'],
            '112': ['104', '111'],
            '113': ['106', 'tollgate1'],
            '114': ['119', '118'],
            '115': ['C', '102'],
            '116': ['118','103','101'],
            '117': ['120', 'tollgate2'],
            '118': ['114', '116', '122'],
            '119': ['108', '114'],
            '120': ['108', '117'],
            '121': ['101', '106'],
            '122': ['118', '103', 'tollgate3'],
            '123': ['110', '107'],

            # nodes
            'A': ['tollgate1', 'tollgate3', '110'],
            'B': ['tollgate1', 'tollgate3', '105'],
            'C': ['115'],
            'tollgate1': ['113', 'A', 'B'],
            'tollgate2': ['117'],
            'tollgate3': ['122', 'A', 'B']}

# Node Types
# '0': normal link
# '1': link, with in_top of NaN
# '2': link, with out_top of NaN
# '3': intersection
# '4': tollgate
# The nodes which have the same types should share parameters
cfg.model.node_type = {# links
             '100': '0',
             '101': '0',
             '102': '0',
             '103': '0',
             '104': '0',
             '105': '0',
             '106': '0',
             '107': '0',
             '108': '0',
             '109': '0',
             '110': '0',
             '111': '0',
             '112': '0',
             '113': '0',
             '114': '0',
             '115': '0',
             '116': '0',
             '117': '0',
             '118': '0',
             '119': '0',
             '120': '0',
             '121': '0',
             '122': '0',
             '123': '0',
              # nodes
             'A': '1',
             'B': '2',
             'C': '3',
             'tollgate1':'4',
             'tollgate2':'5',
             'tollgate3':'6'}



# output
cfg.model.task1_output = {'A': ['tollgate2', 'tollgate3'],
                          'B': ['tollgate1', 'tollgate3'],
                          'C': ['tollgate1', 'tollgate3']}

cfg.model.task2_output = {'tollgate1': 2,
                          'tollgate2': 1,
                          'tollgate3': 2}

# route
cfg.model.route = {'A': [['A', 110, 123, 107, 108, 120, 117,'tollgate2'],
                         ['A', 110, 123, 107, 108, 119, 114, 118, 122, 'tollgate3']],
                   'B': [['B', 105, 100, 111, 103, 116, 101, 121, 106, 113, 'tollgate1'],
                         ['B', 105, 100, 111, 103, 122, 'tollgate3']],
                   'C': [['C', 115, 102, 109, 104, 112, 111, 103, 116, 101, 121, 106, 113, 'tollgate1'],
                         ['C', 115, 102, 109, 104, 112, 111, 103, 122, 'tollgate3']]}

