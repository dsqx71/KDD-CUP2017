from collections import namedtuple
from datetime import datetime, timedelta
from random import shuffle
from config import cfg
import numpy as np
import pandas as pd

DataBatch = namedtuple('DataBatch', ['data', 'label','aux'])

class DataLoader(object):
    
    def __init__(self, data, label, batchsize, time, mode, total_epoch=None): 
        # setting
        data = data.fillna(0)
        label = label.fillna(-1)
        self.data_num = len(data)
        self.batchsize = batchsize
        self.mode = mode
        self.data = {}
        self.label = {}
        self.total_epoch = total_epoch
        self.current_epoch = 0

        # input data
        for key in list(cfg.model.link.keys()) + ['weather']:
            index = [item for item in data.minor_axis if item.startswith(key)]
            if key =='weather':
                continue
                # self.data[key] = data.loc[:, : , index].values
            else:
                self.data[key] = data.loc[:, :5, index].values
        # label 
        for key in label:
            self.label[key] = label[key].values

        # time indicate which time window the data belong to
        self.time = []
        for item in time:
            tmp = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
            self.time.append((tmp.hour*60+tmp.minute) // cfg.time.time_interval)
        self.time = np.array(self.time)
        print (self.time)
        self.original_time = np.array(time)
        # num of data depend on mode
        if  mode == 'test' or mode =='validation':
            self.list_index = [item for item in range(self.data_num) \
                               if self.time[item] == (6*60//cfg.time.time_interval) or self.time[item] == (15*60//cfg.time.time_interval)]
        # elif  mode == 'validation':
        #     self.list_index = [item for item in range(self.data_num) \
        #                        if (self.time[item] >= (5*60//cfg.time.time_interval) and self.time[item] <= (7*60//cfg.time.time_interval)) or  
        #                           (self.time[item] >= (14*60//cfg.time.time_interval) and self.time[item] <= (16*60//cfg.time.time_interval))]
        elif mode == 'train':
            self.list_index1 = [item for item in range(self.data_num)]
            self.list_index2 = [item for item in range(self.data_num) \
                               if self.time[item] == (6*60//cfg.time.time_interval) or self.time[item] == (15*60//cfg.time.time_interval)]
            self.list_index = self.list_index1 + self.list_index2 * 50
        else:
            raise ValueError("The mode doesn't exist")        
        self.data_num = len(self.list_index)
        if mode == 'train':
            self.loss_scale = np.array(cfg.model.loss_scale)
        print (self.data_num)
        self.reset()

    def __iter__(self):
        return self
    
    def reset(self):
        self.current_epoch +=1
        if self.mode == 'train':
            # if self.current_epoch < 5:
            #     self.list_index = self.list_index2
            # else:
            self.list_index = self.list_index1
            self.data_num = len(self.list_index)
            shuffle(self.list_index)
        self.index = 0
    
    def getdata(self, list_index):
        data = {}
        if self.mode == 'train':
            data['loss_scale:0'] = []
            for index in list_index:
                data['loss_scale:0'].append(self.loss_scale[index+6:index+12])
            data['loss_scale:0'] = np.array(data['loss_scale:0']) * np.array([1,1,1,1,1,1])
        for key in self.data:
            data[key + ':0'] = self.data[key][list_index]
        data['time'+':0'] = self.time[list_index]
        return data
    
    def getlabel(self, list_index):
        if self.mode == 'test':
            return None
        label = {}
        for key in self.label:
            # if 'A' in key or 'B' in key or 'C' in key:
            label[key+':0'] = self.label[key][list_index]
        return label
    
    def __next__(self):
        if self.index + self.batchsize > self.data_num:
            raise StopIteration
        else:
            current_index = self.list_index[self.index: self.index + self.batchsize]
            aux = self.list_index[self.index]
            self.index += self.batchsize
            return DataBatch(data=self.getdata(current_index), 
                             label=self.getlabel(current_index),
                             aux=aux)