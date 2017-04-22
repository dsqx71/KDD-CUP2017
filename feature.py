import os
import logging
import pandas as pd

from util import GetTimeslot
from datetime import datetime, timedelta
from numba import jit
from config import cfg
from util import ReadRawdata

def TrajectoryBaiscFeature(trajectory):
    
    logging.info("Extracting basic features from trajectory rawdata...")
    trajectory_feature = {}

    for time in cfg.time.all_timeslots:

        start_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        end_time = start_time +  timedelta(hours=4)

        # data belong to this interval
        mask = (trajectory['starting_time'] >= start_time) & (trajectory['starting_time'] < end_time)
        data = trajectory.loc[mask]

        trajectory_feature[time] = {}
        now = start_time

        for step in range(4*60//cfg.time.time_interval):
            trajectory_feature[time][now.strftime("%Y-%m-%d %H:%M:%S")] = {}
            now = now + timedelta(minutes=cfg.time.time_interval)

        for i in range(len(data)):
            
            # columns : "intersection_id","tollgate_id","vehicle_id","starting_time","travel_seq","travel_time"
            intersection = data.iat[i, 0] + '_'
            tollgate = "tollgate{}".format(data.iat[i, 1]) + '_'
            starting_time = data.iat[i, 3]
            travel_time = float(data.iat[i, 5])

            # intersection feature and label
            time_slot = GetTimeslot(starting_time)
            trajectory_feature[time][time_slot][intersection + 'total_num'] = \
                trajectory_feature[time][time_slot].get(intersection + 'total_num', 0) + 1
            trajectory_feature[time][time_slot][intersection + 'total_time'] = \
                trajectory_feature[time][time_slot].get(intersection + 'total_time', 0) + travel_time

            trajectory_feature[time][time_slot][intersection + 'avg_time'] = \
                trajectory_feature[time][time_slot][intersection + 'total_time'] / trajectory_feature[time][time_slot][intersection + 'total_num']

            trajectory_feature[time][time_slot][intersection + 'num_{}'.format(tollgate)] = \
                trajectory_feature[time][time_slot].get(intersection + 'num_{}'.format(tollgate), 0) + 1
            trajectory_feature[time][time_slot][intersection + 'time_{}'.format(tollgate)] = \
                trajectory_feature[time][time_slot].get(intersection + 'time_{}'.format(tollgate), 0) + travel_time

            trajectory_feature[time][time_slot][intersection + 'avg_time_{}'.format(tollgate)] = \
                trajectory_feature[time][time_slot][intersection + 'time_{}'.format(tollgate)] / \
                trajectory_feature[time][time_slot][intersection + 'num_{}'.format(tollgate)]

            trajectory_feature[time][time_slot][intersection + 'datamiss_{}'.format(tollgate)] = 0

            if starting_time < start_time + timedelta(hours=2):

                # tollgate feature
                arrive_time = starting_time + timedelta(minutes=travel_time/60)
                for item in [(arrive_time,'arrive')]:
                    if item[0] >= start_time and item[0] < end_time:
                        time_slot = GetTimeslot(item[0])
                        trajectory_feature[time][time_slot][tollgate + 'total_num_{}'.format(item[1])] = \
                            trajectory_feature[time][time_slot].get(tollgate + 'total_num_{}'.format(item[1]), 0) + 1
                        trajectory_feature[time][time_slot][tollgate + 'total_time_{}'.format(item[1])] = \
                            trajectory_feature[time][time_slot].get(tollgate+'total_time_{}'.format(item[1]), 0) + travel_time

                        trajectory_feature[time][time_slot][tollgate + 'avg_time_{}'.format(item[1])] = \
                            trajectory_feature[time][time_slot][tollgate + 'total_time_{}'.format(item[1])] / \
                            trajectory_feature[time][time_slot][tollgate + 'total_num_{}'.format(item[1])]

                        trajectory_feature[time][time_slot][tollgate + 'num_{}_{}'.format(intersection, item[1])] = \
                            trajectory_feature[time][time_slot].get(tollgate + 'num_{}_{}'.format(intersection, item[1]), 0) + 1

                        trajectory_feature[time][time_slot][tollgate + 'time_{}_{}'.format(intersection, item[1])] = \
                            trajectory_feature[time][time_slot].get(tollgate + 'time_{}_{}'.format(intersection, item[1]), 0) + travel_time

                        trajectory_feature[time][time_slot][tollgate + 'avg_time_{}_{}'.format(intersection, item[1])] = \
                            trajectory_feature[time][time_slot][tollgate + 'time_{}_{}'.format(intersection, item[1])] / \
                            trajectory_feature[time][time_slot][tollgate + 'num_{}_{}'.format(intersection, item[1])]

                        trajectory_feature[time][time_slot][tollgate + 'data_miss_{}_{}'.format(item[1], intersection)] = 0

                # link feature
                for j in data.iat[i, 4].split(';'):
                    link_name, enter_time, travel_time = j.split('#')
                    link_name = link_name + '_'
                    travel_time = float(travel_time)
                    enter_time = datetime.strptime(enter_time, "%Y-%m-%d %H:%M:%S")
                    for item in [(enter_time, 'enter_time')]:
                        if  item[0]>=start_time and item[0]< end_time:
                            time_slot = GetTimeslot(item[0])
                            trajectory_feature[time][time_slot][link_name+'num_{}'.format(item[1])]  = \
                                trajectory_feature[time][time_slot].get(link_name+'num_{}'.format(item[1]), 0) + 1
                            trajectory_feature[time][time_slot][link_name+'time_{}'.format(item[1])] = \
                                trajectory_feature[time][time_slot].get(link_name+'time_{}'.format(item[1]),0) + travel_time
                            trajectory_feature[time][time_slot][link_name + 'avg_time_{}'.format(item[1])] = \
                                trajectory_feature[time][time_slot][link_name + 'time_{}'.format(item[1])] / \
                                trajectory_feature[time][time_slot][link_name + 'num_{}'.format(item[1])]
                            trajectory_feature[time][time_slot][link_name + 'data_miss_{}'.format(item[1])] = 0

    # convert to Pandas DataFrame
    index_max = 0
    for key in trajectory_feature:
        if pd.DataFrame(trajectory_feature[key]).shape[0] > index_max:
            index_max = pd.DataFrame(trajectory_feature[key]).shape[0]
            index = pd.DataFrame(trajectory_feature[key]).index

    dataframe = {}
    for key in trajectory_feature:
        dataframe[key] = pd.DataFrame(trajectory_feature[key], index=index).T
    
    return dataframe

@jit
def VolumeBasicFeature(data):

    logging.info("Extracting basic features from volume rawdata...")
    
    volume_feature = {}
    for slot in cfg.time.all_timeslots:
        volume_feature[slot] = {}

    # tollgate feature
    for i in range(len(data)):
        # columns : time tollgate_id direction vehicle_model has_etc vehicle_type
        time = data.iat[i, 0]
        tollgate = 'tollgate{}'.format(data.iat[i, 1]) + '_volumn'
        direction = data.iat[i, 2]
        vehicle_model = data.iat[i, 3]
        has_etc = data.iat[i, 4]
        vehicle_type = data.iat[i, 5]
        
        time_slot = GetTimeslot(time)
        volume_feature[time_slot][tollgate + 'num_direction:{}'.format(direction)] = \
            volume_feature[time_slot].get(tollgate + 'num_direction:{}'.format(direction),0) + 1
        volume_feature[time_slot][tollgate + 'num_vehicle_model:{}'.format(vehicle_model)] = \
            volume_feature[time_slot].get(tollgate + 'num_vehicle_model:{}'.format(vehicle_model), 0) + 1
        volume_feature[time_slot][tollgate + 'num_has_etc:{}'.format(has_etc)] = \
            volume_feature[time_slot].get(tollgate + 'num_has_etc:{}'.format(has_etc), 0) + 1
        volume_feature[time_slot][tollgate + 'num_vehicle_type:{}'.format(vehicle_type)] = \
            volume_feature[time_slot].get(tollgate + 'num_vehicle_type:{}'.format(vehicle_type), 0) + 1
        volume_feature[time_slot][tollgate + 'num'] = volume_feature[time_slot].get(tollgate + 'num', 0) + 1
        volume_feature[time_slot][tollgate + 'data_miss'] = 0

    dataframe = pd.DataFrame(volume_feature).T
    return dataframe

@jit
def WeatherBasicFeature(weather, time_interval=cfg.time.time_interval):

    logging.info("Extracting basic features from weather rawdata...")
    data = weather

    weather_feature = {}
    for slot in cfg.time.all_timeslots:
        weather_feature[slot] = {}

    for i in range(len(data)):
        date = data.loc[i]['date']
        hour = data.loc[i]['hour']
        time = datetime.strptime("%s %02d:00:00" % (date, hour), "%Y-%m-%d %H:%M:%S")
        for j in range(int(3 * 60 / time_interval)):
            time_slot = GetTimeslot(time, time_interval)
            weather_feature[time_slot] = data.loc[i].drop('date').astype(float).to_dict()
            time = time + timedelta(minutes=time_interval)
    keys = weather_feature.keys()
    assert len(set(keys)) == len(keys)

    dataframe = pd.DataFrame(weather_feature).T

    return dataframe

@jit
def LinkBasicFeature(link):

    logging.info("Extracting basic features from link rawdata...")

    link['link_id'] = link['link_id'].astype(str)
    link.set_index(['link_id'], inplace=True)
    link.drop(['in_top','out_top'], axis=1, inplace=True)

    link_feature = {}
    for column in link.columns:
        for index in link.index:
            link_feature['{}_{}'.format(index, column)] = link[column][index]
    link_feature = pd.Series(link_feature)

    return link_feature

@jit
def PreprocessingRawdata(update_feature=False):

    logging.info("Started to prepare data...")

    # file path
    data_file = os.path.join(cfg.data.feature_dir, 'basic_feature.pkl')

    # Load existing files
    if os.path.exists(data_file) and update_feature is False:
        logging.info("Loading basic data from {}".format(data_file))
        data = pd.read_pickle(data_file)
    else:
        trajectory, volume, weather, link, route = ReadRawdata()
        
        trajectory_feature = TrajectoryBaiscFeature(trajectory)
        volume_feature = VolumeBasicFeature(volume)
        weather_feature = WeatherBasicFeature(weather)
        link_feature = LinkBasicFeature(link)

        # concat all features
        data = trajectory_feature.copy()
        for time in data:
            timeslots = data[time].index
            for index in link_feature.index:
                data[time][index] = link_feature[index]
                
            data[time] = pd.concat([data[time], weather_feature.loc[timeslots], volume_feature.loc[timeslots]], axis=1).reset_index(drop=True)

        data.to_pickle(data_file)

    return data

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
    logging.info("Combine all basic data...")
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

    Returns
    ----------
    data : dict of numpy array
    """
    logging.info("Filling missing data...")
    
    for key in data:
        data[key] = data[key].fillna(-1)
    return data

def GetLabels(data):
    """
    Parameters
    ----------
    data : dict of Pandas DataFrame

    Returns
    -------
    labels : dict of Pandas Series
    """
    labels = {}

    for intersection in cfg.model.task1_output:
        for tollgate in cfg.model.task1_output[intersection]:
            labels['{}_{}'.format(intersection, tollgate)] = (data[intersection]['time_{}'.format(tollgate)] /
                                                              data[intersection]['num_{}'.format(tollgate)])

    for tollgate in cfg.model.task2_output:
        for direction in range(cfg.model.task2_output[tollgate]):
            labels['{}_{}'.format(tollgate, direction)] = data[tollgate]['volume_num_direction:{}'.format(direction)]
    return labels

def SplitData(data, label):
    """
    Split Data into training set, validation set, and testing set
    """
    data_train = {}
    data_validation = {}
    data_test = {}

    label_train = {}
    label_validation = {}
    label_test = {}

    for node in data:
        data_train[node] = data[node].loc[cfg.time.train_timeslots]
        data_validation[node] = data[node].loc[cfg.time.validation_timeslots]
        data_test[node] = data[node].loc[cfg.time.test_timeslots]

    for node in label:
        label_train[node] = label[node].loc[cfg.time.train_timeslots]
        label_validation[node] = label[node].loc[cfg.time.validation_timeslots]
        label_test[node] = label[node].loc[cfg.time.test_timeslots]

    return data_train, data_validation, data_test, label_train, label_validation, label_test

def Standardize(data):
    """
    subduce mean and div std
    """
    for node in data:
        for key in data[node]:
            if data[node][key].std() > 0:
                data[node][key] = (data[node][key] - data[node][key].mean()) / data[node][key].std()
    return data