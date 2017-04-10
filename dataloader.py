from collections import namedtuple
from random import shuffle
DataBatch = namedtuple('DataBatch', ['data', 'label','aux'])

class DataLoader(object):
    
    def __init__(self, data, label, batchsize, time, is_train=True):
        # setting
        self.data = data
        self.label = label
        self.batchsize = batchsize
        self.time = np.array(time)
        self.is_train = is_train
        self.num_slots = 2 * 60 // cfg.time.time_interval
        
        # data index
        self.index = 0
        if is_train:
            self.data_num = len(data['100']) - 4 * 60 // cfg.time.time_interval + 1 
        else:
            self.data_num = len(data['100']) - 2 * 60 // cfg.time.time_interval + 1
            
        self.list_index = [item for item in range(self.data_num)]
        
        self.reset()
     
    def __iter__(self):
        return self
    
    def reset(self):
        if self.is_train:
            shuffle(self.list_index)
        self.index = 0
    
    def getdata(self, list_index):    
        data = {key:[] for key in self.data}
        for key in self.data:
            for index in list_index:
                if key != 'weather':
                    data[key].append(self.data[key][index: index + self.num_slots])
                else:
                    data[key].append(self.data[key][index: index + self.num_slots*2])
            data[key] = np.array(data[key])
        data['time'] = self.time[list_index]
        return data
    
    def getlabel(self, list_index):
        if self.is_train == False:
            return None
        
        label = {}
        for key in self.label:
            label[key] = []
            for index in list_index:
                data = self.label[key][index+self.num_slots : index + 2*self.num_slots]
                label[key].append(data)
            label[key] = np.array(label[key])
        
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