from collections import namedtuple
from datetime import datetime, timedelta
from random import shuffle
from config import cfg
import numpy as np

DataBatch = namedtuple('DataBatch', ['data', 'label','aux'])

class DataLoader(object):
    
    def __init__(self, data, label, batchsize, time, mode):
        
        # setting
        self.data = {key:data[key].values for key in data}
        self.label = {key:label[key].values for key in label}

        self.batchsize = batchsize
        self.mode = mode

        # num of slot in encoder or decoder
        self.num_slots = 2 * 60 // cfg.time.time_interval
        
        # self.time denote which time window the data belong to
        self.time = []
        for item in time:
            tmp = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
            self.time.append((tmp.hour*60+tmp.minute) // cfg.time.time_interval)
        self.time = np.array(self.time)
        
        # 
        if mode == 'train' or mode == 'validation':
            self.data_num = len(data['100']) - 4 * 60 // cfg.time.time_interval + 1 
        elif mode == 'test':
            self.data_num = len(data['100']) - 2 * 60 // cfg.time.time_interval + 1
        else:
            raise ValueError("The mode doesn't exist")
        
        if mode == 'validation' or mode == 'test':
            self.list_index = [item for item in range(self.data_num) \
                               if self.time[item] == (6*60//cfg.time.time_interval) or self.time[item] == (15*60//cfg.time.time_interval)]
            self.data_num = len(self.list_index)
        else:
            self.list_index = [item for item in range(self.data_num)]

        if mode == 'train':
            self.loss_scale = cfg.model.loss_scale
        
        self.reset()
     
    def __iter__(self):
        return self
    
    def reset(self):
        if self.mode == 'train':
            shuffle(self.list_index)
        self.index = 0
    
    def getdata(self, list_index):    
        data = {}
        for key in self.data:
            data[key+':0'] = []
            for index in list_index:
                if key != 'weather':
                    data[key+':0'].append(self.data[key][index: index + self.num_slots])
                else:
                    data[key+':0'].append(self.data[key][index: index + self.num_slots*2])
            data[key+':0'] = np.array(data[key+':0'])

        data['time'+':0'] = self.time[list_index]
        if self.mode == 'train':
            data['loss_scale:0'] = []
            for index in list_index:
                data['loss_scale:0'].append(self.loss_scale[index + self.num_slots: index + self.num_slots * 2])
            data['loss_scale:0'] = np.array(data['loss_scale:0'])

        return data
    
    def getlabel(self, list_index):
        
        if self.mode == 'test':
            return None
        
        label = {}
        for key in self.label:
            if 'A' in key or 'B' in key or 'C' in key:
                label[key+':0'] = []
                for index in list_index:
                    data = self.label[key][index+self.num_slots : index + 2*self.num_slots]
                    label[key+':0'].append(data)
                label[key+':0'] = np.array(label[key+':0'])
        
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