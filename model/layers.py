import tensorflow as tf
import math
import numpy as np

from tensorflow.contrib.rnn import RNNCell
from config import cfg

def Graph_Convolution(data, out_dim, name):
    in_dim = data[0].get_shape()[1]
    result = []
    W = tf.get_variable(name=name+'_weight', shape=[in_dim, out_dim])
    for index, datum in enumerate(data):
        tmp = tf.matmul(datum, W)
        if index == 0:
            result = tmp
        else:
            result = result + tmp
    result = result / len(data)
    result = tf.nn.relu(result)
    return result


def GC(data, out_dim, name):

    in_dim = data['100'].get_shape()[1].value
    result = {}
    for node in cfg.model.link:
        W = tf.get_variable(name=name+'{}_weight'.format(node), shape=[in_dim, out_dim])
        for i in range(len(cfg.model.link[node])):
            # W = tf.get_variable(name=name+'{}_{}_weight'.format(node,i), shape=[in_dim, out_dim])
            if i == 0:
                result[node] = tf.matmul(data[cfg.model.link[node][i]], W)
            else:
                result[node] = result[node] + tf.matmul(data[cfg.model.link[node][i]], W)
        # result[node] =  tf.nn.sigmoid(result[node])
    return result
        # result[node] = tf.nn.relu(result[node] / len(cfg.model.link[node]))
    # print (result['100'].get_shape()[1].value)
    

def FC(x, in_dim, out_dim, name, activation='relu', is_training=True, with_bn=False):
    """
    Fully connect
    """
    W = tf.get_variable(name=name+'_weight', shape=[in_dim, out_dim]) 
    b = tf.get_variable(name=name+'_bias', shape=[out_dim], initializer=tf.zeros_initializer())

    y = tf.matmul(x, W) + b

    if with_bn:
        y = tf.contrib.layers.batch_norm(y, 
                                        decay=0.95,
                                        epsilon=1E-5,
                                        center=True, 
                                        scale=True, 
                                        fused=False,
                                        is_training=is_training,
                                        scope=name+'_bn')
    if activation == 'relu':
        y = tf.nn.relu(y, name=name + '_relu')
    
    elif activation == 'softmax':
        y = tf.nn.softmax(y, name=name + '_softmax')
    
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(y, name=name + '_sigmoid')

    return y

def MultiLayerFC(name, data, in_dim, out_dim, num_hidden, num_layer, activation='relu', is_training=True):
    
    if num_layer > 1:
        data = FC(data, in_dim=in_dim, out_dim=num_hidden, name='{}_fc0'.format(name), 
                activation=activation, is_training=is_training)
        
        for i in range(1, num_layer-1):
             data= FC(data, in_dim = num_hidden, out_dim = num_hidden, name = '{}_fc{}'.format(name, i), 
                activation=activation, is_training=is_training)
        
        data = FC(data, in_dim=num_hidden, out_dim=out_dim, name='{}_fc{}'.format(name, num_layer-1), 
            activation=activation, is_training=is_training)
    
    elif num_layer == 1:   
        data = FC(data, in_dim=in_dim, out_dim=out_dim, name='{}_fc0'.format(name), 
            activation=activation, is_training=is_training)
    
    return data

def shortcut(data, out_dim, name, dropout_prob=None):
    
    if dropout_prob is not None:  
        data = tf.nn.dropout(data, keep_prob=1 - dropout_prob)
    data0 = data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = out_dim, name=name + 'fc1')
    data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = out_dim, name=name + 'fc2')
    data = data + data0
    
    return data

class BNLSTMCell(RNNCell):
    """
    Recurrent Batch Normalization
    """
    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units])
            
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units])

            bias = tf.get_variable('bias', [4 * self.num_units], 
                initializer=tf.constant_initializer(0, dtype=tf.float32))

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = tf.contrib.layers.batch_norm(xh, 
                                                decay=0.99,
                                                epsilon=1E-3,
                                                center=True,
                                                scale=True,
                                                fused=True,
                                                is_training=self.training,
                                                scope='xh')

            bn_hh = tf.contrib.layers.batch_norm(hh, 
                                                decay=0.99,
                                                epsilon=1E-3,
                                                center=True,
                                                scale=True,
                                                fused=True,
                                                is_training=self.training,
                                                scope='hh')

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(axis=1, num_or_size_splits = 4, value = hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = tf.contrib.layers.batch_norm(new_c,
                                                decay=0.99,
                                                epsilon=1E-3,
                                                center=True,
                                                scale=True,
                                                fused=True,
                                                is_training=self.training,
                                                scope='c')

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)