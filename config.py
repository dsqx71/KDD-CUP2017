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

# testing timeslots
cfg.time.test_timeslots = []
for i in range(18, 25):
    for k in ['06', '15']:
        left = datetime.strptime("2016-10-{} {}:00:00".format(i, k), "%Y-%m-%d %H:%M:%S")
        for j in range(int(2*60/cfg.time.time_interval)):
            cfg.time.test_timeslots.append(left.strftime("%Y-%m-%d %H:%M:%S"))
            left = left + timedelta(minutes=cfg.time.time_interval)

# training timeslots
cfg.time.train_timeslots = []
right = cfg.time.trajectory_train_start
for slot in range(int(cfg.time.trajectory_slots)):
    left = right
    right = left + timedelta(minutes=cfg.time.time_interval)
    cfg.time.train_timeslots.append(left.strftime("%Y-%m-%d %H:%M:%S"))

cfg.time.all_timeslots = cfg.time.train_timeslots.copy()
cfg.time.all_timeslots.extend(cfg.time.test_timeslots)

# padding timeslots
for i in range(18, 25):
    left = datetime.strptime("2016-10-{} 00:00:00".format(i, k), "%Y-%m-%d %H:%M:%S")
    for j in range(int(24*60/cfg.time.time_interval)):
        time = left.strftime("%Y-%m-%d %H:%M:%S")
        left = left + timedelta(minutes=cfg.time.time_interval)
        if time not in cfg.time.all_timeslots:
            cfg.time.all_timeslots.append(time)

cfg.time.train_timeslots.sort()
cfg.time.test_timeslots.sort()
cfg.time.all_timeslots.sort()
assert len(np.unique(np.array(cfg.time.all_timeslots))) == len(cfg.time.all_timeslots), 'Time slots not unique'


#### Model

# RNN topology
cfg.model.link = {# links
            '100': ['105'],
            '101': ['116'],
            '102': ['115'],
            '103': ['111'],
            '104': ['109'],
            '105': ['B'],
            '106': ['121'],
            '107': ['123'],
            '108': ['107'],
            '109': ['102'],
            '110': ['A'],
            '111': ['100', '112'],
            '112': ['104'],
            '113': ['106'],
            '114': ['119'],
            '115': ['C'],
            '116': ['118','103'],
            '117': ['120'],
            '118': ['114'],
            '119': ['108'],
            '120': ['108'],
            '121': ['101'],
            '122': ['118','103'],
            '123': ['110'],

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
cfg.model.route = {'A': [[110, 123, 107, 108, 120, 117],
                        [110, 123, 107, 108, 119, 114, 118, 122]],
                   'B': [[105, 100, 111, 103, 116, 101, 121, 106, 113],
                        [105, 100, 111, 103, 122]],
                   'C': [[115, 102, 109, 104, 112, 111, 103, 116, 101, 121, 106, 113],
                        [115, 102, 109, 104, 112, 111, 103, 122]]}

