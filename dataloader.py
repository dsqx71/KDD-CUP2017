from random import shuffle

DataBatch = namedtuple('DataBatch', ['data', 'label', 'index'])
class DataLoader():
    
    def __init__(data, label, batchsize, time, is_train=True, is_shuffle=True):
        # setting
        
        self.data = data
        self.label = label
        self.batchsize = batchsize
        self.time = time
        self.is_shuffle = is_shuffle
        self.is_train = is_train
        
        # data index
        self.index = 0
        self.data_num = len(data['100']) - 4 * 60 // cfg.time.time_interval + 1 
        self.list_index = [item for item in range(self.data_num)]
        
        self.reset()
            
    def __iter__(self):
        
        return self
    
    def reset(self):
        if self.is_shuffle:
            shuffle(self.list_index)
        self.index = 0
    
    def get_data(self, current_index):

        data = {key:[] for key in self.data}

    def next(self):
        if self.index + self.batchsize >= self.epoch_size:
            raise StopIteration
        
        data = {key:[] for key in self.data}
        
        current_index = self.list_index[index:index+self.batchsize]
        for key in self.data:
            data[key].append(self.data[key][current_index])
        
            