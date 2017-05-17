import math, json, pandas as pd
import os
from datetime import datetime, timedelta
from config import cfg

def GetTimeslot(time, interval=cfg.time.time_interval):
    """
    Parameters
    ----------
    time : datetime,
    interval : float
        time interval

    Returns
    -------
    time_slot :  str,
        denote which time window the data belong to
    """
    minute = math.floor(time.minute / interval) * interval
    new_time = datetime(time.year, time.month, time.day, time.hour, minute, 0)
    return new_time.strftime("%Y-%m-%d %H:%M:%S")

def ReadJson(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def WriteJson(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def ReadRawdata():
    """
    read Rawdata

    Returns
    -------
    trajectory, volume, weather, link, route : pandas.DataFrame
    """
    # time-independent data
    rawdata_dir = cfg.data.rawdata_dir

    link = pd.read_csv(os.path.join(rawdata_dir, 'training/links (table 3).csv'))
    route = pd.read_csv(os.path.join(rawdata_dir, 'training/routes (table 4).csv'))

    # training rawdata
    trajectory_train = pd.read_csv(os.path.join(rawdata_dir, 'training/trajectories(table 5)_training.csv'))
    volume_train = pd.read_csv(os.path.join(rawdata_dir, 'training/volume(table 6)_training.csv'))
    weather_train = pd.read_csv(os.path.join(rawdata_dir, 'training/weather (table 7)_training.csv'))

    # testing rawdata
    trajectory_test = pd.read_csv(os.path.join(rawdata_dir, 'testing_phase1/trajectories(table 5)_test1.csv'))
    volume_test = pd.read_csv(os.path.join(rawdata_dir, 'testing_phase1/volume(table 6)_test1.csv'))
    weather_test = pd.read_csv(os.path.join(rawdata_dir, 'testing_phase1/weather (table 7)_test1.csv'))

    # concat train and test
    trajectory = pd.concat([trajectory_train, trajectory_test], ignore_index=True)
    volume = pd.concat([volume_train, volume_test], ignore_index=True)
    weather = pd.concat([weather_train, weather_test], ignore_index=True)
    weather['hour'] = weather['hour'].astype(float)

    # convert str to datetime
    trajectory['starting_time'] = trajectory['starting_time'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    volume['time'] = volume['time'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    return trajectory, volume, weather, link, route


def GetDataNum(num):

    return num - 4 * 60 // cfg.time.time_interval + 1

def GetTotalSecond(date):

    return (date - datetime(2000, 1, 1)).total_seconds()
