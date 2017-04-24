from collections import namedtuple
from datetime import datetime, timedelta
from random import shuffle
from config import cfg
import numpy as np
import pandas as pd

DataBatch = namedtuple('DataBatch', ['data', 'label','aux'])

class DataLoader(object):
    
    def __init__(self, data, label, batchsize, time, mode): 
        # setting
        self.data = data
        self.label = label
        self.batchsize = batchsize
        self.mode = mode
        self.data_num = len(data)
        self.num_slots = 2 * 60 // cfg.time.time_interval
        self.feature_name = list(data.minor_axis)
        self.feature_name.sort()
        
        self.time = []
        for item in time:
            tmp = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
            self.time.append((tmp.hour*60+tmp.minute) // cfg.time.time_interval)
        self.time = pd.Series(self.time, index=time)
           
        if mode == 'validation' or mode == 'test':
            self.list_index = [time[item] for item in range(self.data_num) \
                               if self.time[item] == (6*60//cfg.time.time_interval) or self.time[item] == (15*60//cfg.time.time_interval)]
            self.data_num = len(self.list_index)
        elif mode == 'train':
            self.list_index = [time[item] for item in range(self.data_num)]
        else:
            raise ValueError("The mode doesn't exist")        
        self.reset()
     
    def __iter__(self):
        return self
    
    def reset(self):
        if self.mode == 'train':
            shuffle(self.list_index)
        self.index = 0
    
    def getdata(self, list_index):    
        data = {}
        for key in cfg.model.link:
            mirror_items = [name for name in self.feature_name if name.startswith(key)]
            data[key + ':0'] = self.data.loc[list_index, :5, mirror_items]
        data['time'+':0'] = self.time[list_index]
        return data
    
    def getlabel(self, list_index):
        
        if self.mode == 'test':
            return None

        label = {}
        for key in self.label:
            if 'A' in key or 'B' in key or 'C' in key:
                label[key+':0'] = self.label[key].loc[list_index]
        return label
    
    @property        
    def getindex(self):
        return self.index
    
    def __next__(self):
        if self.index + self.batchsize >= self.data_num:
            raise StopIteration
        else:
            current_index = self.list_index[self.index: self.index + self.batchsize]
            self.index += self.batchsize
            return DataBatch(data=self.getdata(current_index), 
                             label=self.getlabel(current_index),
                             aux=None)