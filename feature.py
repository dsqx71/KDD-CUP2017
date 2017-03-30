from util import GetTimeslot
from datetime import datetime, timedelta
from numba import jit
import os
import platform
import pandas as pd
import multiprocessing as mp
from config import cfg
from util import ReadJson, WriteJson, ReadRawdata

@jit
def ExtractTrajectoryRawdata(data, trajectory_feature, time_interval):

    for i in range(len(data)):

        intersection = data.loc[i]['intersection_id']
        tollgate = "tollgate{}".format(data.loc[i]['tollgate_id'])
        starting_time = data.loc[i]['starting_time']
        travel_time = float(data.loc[i]['travel_time'])
        arrive_time = starting_time + timedelta(minutes=travel_time)

        # intersection feature
        time_slot = GetTimeslot(starting_time, interval=time_interval)
        trajectory_feature[intersection][time_slot]['num'] = \
            trajectory_feature[intersection][time_slot].get('num', 0) + 1
        trajectory_feature[intersection][time_slot]['time'] = \
            trajectory_feature[intersection][time_slot].get('time', 0) + travel_time
        trajectory_feature[intersection][time_slot]['data_miss'] = 0
        # tollgate feature
        time_slot = GetTimeslot(arrive_time, interval=time_interval)
        trajectory_feature[tollgate][time_slot]['num'] = trajectory_feature[tollgate][time_slot].get('num', 0) + 1
        trajectory_feature[tollgate][time_slot]['time'] = trajectory_feature[tollgate][time_slot].get('time', 0) + travel_time
        trajectory_feature[tollgate][time_slot]['data_miss'] = 0
        # link feature
        for j in data.loc[i]['travel_seq'].split(';'):
            link_name, enter_time, travel_time = j.split('#')
            travel_time = float(travel_time)
            enter_time = datetime.strptime(enter_time, "%Y-%m-%d %H:%M:%S")
            time_slot = GetTimeslot(enter_time, interval=time_interval)
            trajectory_feature[link_name][time_slot]['num'] = trajectory_feature[link_name][time_slot].get('num',0)+ 1
            trajectory_feature[link_name][time_slot]['time'] = trajectory_feature[link_name][time_slot].get('num',0) + travel_time
            trajectory_feature[link_name][time_slot]['data_miss'] = 0

@jit
def ExtractVolumeRawdata(data, volume_feature, time_interval):

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
            volume_feature[tollgate][time_slot].get('num_vehicle_model:{}'.format(vehicle_model), 0)+1
        volume_feature[tollgate][time_slot]['num_has_etc:{}'.format(has_etc)] = \
            volume_feature[tollgate][time_slot].get('num_has_etc:{}'.format(has_etc), 0) + 1
        volume_feature[tollgate][time_slot]['num_vehicle_type:{}'.format(vehicle_type)] = \
            volume_feature[tollgate][time_slot].get('num_vehicle_type:{}'.format(vehicle_type), 0) + 1
        volume_feature[tollgate][time_slot]['num'] = volume_feature[tollgate][time_slot].get('num', 0) + 1
        volume_feature[tollgate][time_slot]['data_miss'] = 0

@jit
def ExtractWeatherRawdata(weather, weather_feature, time_interval):

    data = weather
    for i in range(len(data)):
        date = data.loc[i]['date']
        hour = data.loc[i]['hour']
        time = datetime.strptime("%s %02d:00:00" % (date, hour), "%Y-%m-%d %H:%M:%S")
        for j in range(int(3*60/time_interval)):
            time_slot = GetTimeslot(time, time_interval)
            weather_feature[time_slot] = data.loc[i].to_dict()
            time = time + timedelta(minutes=time_interval)
    keys = weather_feature.keys()
    assert len(set(keys)) == len(keys)

@jit
def PreprocessingRawdata(force_update=False):

    # file path
    volume_feature_file = os.path.join(cfg.data.feature_dir, 'volume_feature.json')
    trajectory_feature_file = os.path.join(cfg.data.feature_dir, 'trajectory_feature.json')
    weather_feature_file = os.path.join(cfg.data.feature_dir, 'weather_feature.json')

    # read existing files
    if os.path.exists(volume_feature_file) and \
       os.path.exists(trajectory_feature_file) and \
       os.path.exists(weather_feature_file) and force_update is False:
        volume_feature = ReadJson(volume_feature_file)
        trajectory_feature = ReadJson(trajectory_feature_file)
        weather_feature = ReadJson(weather_feature_file)
    else:
        # init feature dict
        trajectory_feature = {}
        volume_feature = {}
        weather_feature = {}

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

        for slot in cfg.time.all_timeslots:
            weather_feature[slot] = {}

        # Extract features
        trajectory, volume, weather, link, route = ReadRawdata()

        # multiprocessing setting
        if platform.system() == 'Windows':
            mp.freeze_support()

        # Extract feature
        pool = mp.Pool()
        pool.apply_async(ExtractTrajectoryRawdata, (trajectory, trajectory_feature, cfg.time.time_interval))
        pool.apply_async(ExtractVolumeRawdata, (volume, volume_feature, cfg.time.time_interval))
        pool.apply_async(ExtractWeatherRawdata, (weather, weather_feature, cfg.time.time_interval))
        pool.close()
        pool.join()

        # Save features
        WriteJson(volume_feature_file, volume_feature)
        WriteJson(trajectory_feature_file, trajectory_feature)
        WriteJson(weather_feature_file, weather_feature)

    return volume_feature, trajectory_feature, weather_feature

@jit
def ReformatData(volume_feature, trajectory_feature, weather_feature):

    data = trajectory_feature.copy()
    data['weather'] = weather_feature

    for node in volume_feature:
        for timeslot in volume_feature[node]:
            for feature in volume_feature[node][timeslot]:
                data[node][timeslot]['volume_{}'.format(feature)] = volume_feature[node][timeslot][feature]

    for key in data:
        data[key] = pd.DataFrame(data[key]).transpose()

    return data