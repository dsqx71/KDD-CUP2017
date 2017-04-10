from util import GetTimeslot
from datetime import datetime, timedelta
from numba import jit
import os
import logging
import pandas as pd
from config import cfg
from util import ReadJson, WriteJson, ReadRawdata

@jit
def ExtractTrajectoryRawdata(data, trajectory_feature, time_interval=cfg.time.time_interval):

    logging.info("Extracting basic features from trajectroy rawdata...")

    for i in range(len(data)):
        intersection = data.loc[i]['intersection_id']
        tollgate = "tollgate{}".format(data.loc[i]['tollgate_id'])
        starting_time = data.loc[i]['starting_time']
        travel_time = float(data.loc[i]['travel_time'])
        arrive_time = starting_time + timedelta(minutes=travel_time)

        # intersection feature and label
        time_slot = GetTimeslot(starting_time, interval=time_interval)
        trajectory_feature[intersection][time_slot]['total_num'] = \
            trajectory_feature[intersection][time_slot].get('total_num', 0) + 1
        trajectory_feature[intersection][time_slot]['total_time'] = \
            trajectory_feature[intersection][time_slot].get('total_time', 0) + travel_time
        trajectory_feature[intersection][time_slot]['data_miss'] = 0

        trajectory_feature[intersection][time_slot]['num_{}'.format(tollgate)] = \
            trajectory_feature[intersection][time_slot].get('num_{}'.format(tollgate), 0) + 1
        trajectory_feature[intersection][time_slot]['time_{}'.format(tollgate)] = \
            trajectory_feature[intersection][time_slot].get('time_{}'.format(tollgate), 0) + travel_time
        trajectory_feature[intersection][time_slot]['datamiss_{}'.format(tollgate)] = 0

        # tollgate feature
        for time in [(arrive_time,'arrive'), (starting_time, 'start')]:
            time_slot = GetTimeslot(time[0], interval=time_interval)
            trajectory_feature[tollgate][time_slot]['total_num_{}'.format(time[1])] = \
                trajectory_feature[tollgate][time_slot].get('total_num_{}'.format(time[1]), 0) + 1
            trajectory_feature[tollgate][time_slot]['total_time_{}'.format(time[1])] = \
                trajectory_feature[tollgate][time_slot].get('total_time_{}'.format(time[1]), 0) + travel_time
            trajectory_feature[tollgate][time_slot]['data_miss_{}'.format(time[1])] = 0
            trajectory_feature[tollgate][time_slot]['num_{}_{}'.format(intersection, time[1])] = \
                trajectory_feature[tollgate][time_slot].get('num_{}_{}'.format(intersection, time[1]), 0) + 1
            trajectory_feature[tollgate][time_slot]['time_{}_{}'.format(intersection, time[1])] = \
                trajectory_feature[tollgate][time_slot].get('time_{}_{}'.format(intersection, time[1]), 0) + travel_time

        # link feature
        for j in data.loc[i]['travel_seq'].split(';'):
            link_name, enter_time, travel_time = j.split('#')
            travel_time = float(travel_time)
            enter_time = datetime.strptime(enter_time, "%Y-%m-%d %H:%M:%S")
            for time in [(starting_time, 'start'), (enter_time, 'enter_time')]:
                time_slot = GetTimeslot(time[0], interval=time_interval)
                trajectory_feature[link_name][time_slot]['num_{}'.format(time[1])]  = \
                    trajectory_feature[link_name][time_slot].get('num_{}'.format(time[1]), 0) + 1
                trajectory_feature[link_name][time_slot]['time_{}'.format(time[1])] = \
                    trajectory_feature[link_name][time_slot].get('time_{}'.format(time[1]),0) + travel_time
                trajectory_feature[link_name][time_slot]['data_miss_{}'.format(time[1])] = 0

@jit
def ExtractVolumeRawdata(data, volume_feature, time_interval=cfg.time.time_interval):

    logging.info("Extracting basic features from volume rawdata...")
    # tollgate feature
    for i in range(len(data)):
        tollgate = 'tollgate{}'.format(data.loc[i]['tollgate_id'])
        direction = data.loc[i]['direction']
        vehicle_model = data.loc[i]['vehicle_model']
        has_etc = data.loc[i]['has_etc']
        vehicle_type = data.loc[i]['vehicle_type']
        time = data.loc[i]['time']

        time_slot = GetTimeslot(time, time_interval)
        volume_feature[tollgate][time_slot]['num_direction:{}'.format(direction)] = \
            volume_feature[tollgate][time_slot].get('num_direction:{}'.format(direction),0) + 1
        volume_feature[tollgate][time_slot]['num_vehicle_model:{}'.format(vehicle_model)] = \
            volume_feature[tollgate][time_slot].get('num_vehicle_model:{}'.format(vehicle_model), 0) + 1
        volume_feature[tollgate][time_slot]['num_has_etc:{}'.format(has_etc)] = \
            volume_feature[tollgate][time_slot].get('num_has_etc:{}'.format(has_etc), 0) + 1
        volume_feature[tollgate][time_slot]['num_vehicle_type:{}'.format(vehicle_type)] = \
            volume_feature[tollgate][time_slot].get('num_vehicle_type:{}'.format(vehicle_type), 0) + 1
        volume_feature[tollgate][time_slot]['num'] = volume_feature[tollgate][time_slot].get('num', 0) + 1
        volume_feature[tollgate][time_slot]['data_miss'] = 0

@jit
def ExtractWeatherRawdata(weather, weather_feature, time_interval=cfg.time.time_interval):

    logging.info("Extracting basic features from weather rawdata...")
    data = weather
    for i in range(len(data)):
        date = data.loc[i]['date']
        hour = data.loc[i]['hour']
        time = datetime.strptime("%s %02d:00:00" % (date, hour), "%Y-%m-%d %H:%M:%S")
        for j in range(int(3*60/time_interval)):
            time_slot = GetTimeslot(time, time_interval)
            if time_slot in weather_feature:
                weather_feature[time_slot] = data.loc[i].drop('date').astype(float).to_dict()
            time = time + timedelta(minutes=time_interval)
    keys = weather_feature.keys()
    assert len(set(keys)) == len(keys)

@jit
def ExtractLinkRawdata(link, link_feature, time_interval=cfg.time.time_interval):

    logging.info("Extracting basic features from link rawdata...")
    link['link_id'] = link['link_id'].astype(str)
    link.set_index(['link_id'], inplace=True)
    for node in link.index:
        for timeslot in link_feature[node]:
            link_feature[node][timeslot] = link.loc[node].drop(['in_top','out_top']).astype(float).to_dict()

@jit
def PreprocessingRawdata(force_update=False):

    logging.info("Started to prepare data...")

    # file path
    volume_feature_file = os.path.join(cfg.data.feature_dir, 'volume_feature.json')
    trajectory_feature_file = os.path.join(cfg.data.feature_dir, 'trajectory_feature.json')
    weather_feature_file = os.path.join(cfg.data.feature_dir, 'weather_feature.json')
    link_feature_file = os.path.join(cfg.data.feature_dir, 'link_feature.json')

    # read existing files
    if os.path.exists(volume_feature_file) and \
       os.path.exists(trajectory_feature_file) and \
       os.path.exists(weather_feature_file) and \
       os.path.exists(link_feature_file) and force_update is False:
        logging.info("Loading data from existing json files: volume_feature.json, "
                     "trajectory_feature.json, weather_feature.json...")
        volume_feature = ReadJson(volume_feature_file)
        trajectory_feature = ReadJson(trajectory_feature_file)
        weather_feature = ReadJson(weather_feature_file)
        link_feature = ReadJson(link_feature_file)
    else:
        # init feature dict
        trajectory_feature = {}
        volume_feature = {}
        weather_feature = {}
        link_feature = {}

        for node in cfg.model.link:
            # trajectory
            trajectory_feature[node] = {}
            for slot in cfg.time.all_timeslots:
                trajectory_feature[node][slot] = {}

            # volume
            if 'tollgate' in node:
                volume_feature[node] = {}
                for slot in cfg.time.all_timeslots:
                    volume_feature[node][slot] = {}

            link_feature[node] = {}
            for slot in cfg.time.all_timeslots:
                link_feature[node][slot] = {}

        for slot in cfg.time.all_timeslots:
            weather_feature[slot] = {}

        trajectory, volume, weather, link, route = ReadRawdata()

        # Extract feature
        ExtractLinkRawdata(link, link_feature)
        ExtractTrajectoryRawdata(trajectory, trajectory_feature)
        ExtractVolumeRawdata(volume, volume_feature)
        ExtractWeatherRawdata(weather, weather_feature)

        # Save features
        WriteJson(volume_feature_file, volume_feature)
        WriteJson(trajectory_feature_file, trajectory_feature)
        WriteJson(weather_feature_file, weather_feature)
        WriteJson(link_feature_file, link_feature)

    return volume_feature, trajectory_feature, weather_feature, link_feature

@jit
def CombineBasicFeature(volume_feature, trajectory_feature, weather_feature, link_feature):
    """
    Parameters
    ----------
    volume_feature : dict of dict
    trajectory_feature : dict of dict
    weather_feature : dict of dict
    link_feature :  dict of dict

    Returns
    -------
    data : dict of Pandas.DataFrame
    """
    logging.info("Combine all basic data")
    data = trajectory_feature.copy()
    data['weather'] = weather_feature

    for node in volume_feature:
        for timeslot in volume_feature[node]:
            for feature in volume_feature[node][timeslot]:
                data[node][timeslot]['volume_{}'.format(feature)] = volume_feature[node][timeslot][feature]

    for node in data:
        data[node] = pd.DataFrame(data[node]).transpose()

    for node in link_feature:
        data[node] = pd.concat([data[node], pd.DataFrame(link_feature[node]).T], axis='columns')

    return data

def FillingMissingData(data):
    """
    Parameters
    ----------
    data :  dict of Pandas.DataFrame,
    """
    logging.info("Filling missing data...")
    # TODO: Need more experiements to find proper filling values
    for key in data:
        data[key] = data[key].fillna(-1).values
    return data

def GetLabels(data):
    """
    Parameters
    ----------
    data : dict of Pandas DataFrame

    Returns
    -------
    labels : dict
    """
    labels = {}

    for intersection in cfg.model.task1_output:
        for tollgate in cfg.model.task1_output[intersection]:
            labels['{}_{}'.format(intersection, tollgate)] = (data[intersection]['time_{}'.format(tollgate)] /
                                                              data[intersection]['num_{}'.format(tollgate)]).values

    for tollgate in cfg.model.task2_output:
        for direction in range(cfg.model.task2_output[tollgate]):
            labels['{}_{}'.format(tollgate, direction)] = data[tollgate]['volume_num_direction:{}'.format(direction)].values
    return labels

def SplitData(data):

    data_train = {}
    for node in data:
        data_train[node] = data[node].loc[cfg.time.train_timeslots]
    data_test = {}
    for node in data:
        data_test[node] = data[node].loc[cfg.time.test_timeslots]

    return data_train, data_test