import tensorflow as tf
import math
import numpy as np

from tensorflow.contrib.rnn import RNNCell
from config import cfg


def conv(data, name, width, in_dim, out_dim, stride, pad):
    filters = tf.get_variable(name=name+'_weight', shape=[width, in_dim, out_dim])
    data = tf.nn.conv1d(value=data, filters=filters, stride=stride, padding=pad, data_format="NHWC")
    return data

def prelu(_x, name):
  alphas = tf.get_variable(name + '_alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                           dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def Graph_Convolution(data, out_dim, name):
    in_dim = data[list(data.keys())[0]].get_shape()[1].value
    result = {}
    W = tf.get_variable(name=name+'_weight', shape=[in_dim, out_dim])
    for node in cfg.model.link:
        for i in range(len(cfg.model.link[node])):
            if i == 0:
                result[node] = tf.matmul(data[cfg.model.link[node][i]], W)
            else:
                result[node] = result[node] + tf.matmul(data[cfg.model.link[node][i]], W)
        result[node] = prelu(result[node] / len(cfg.model.link[node]), name = name + node + '_prelu')
    return result

def FC(x, in_dim, out_dim, name, activation='prelu', is_training=True, with_bn=False):
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

    elif activation == 'prelu':
        y = prelu(y, name = name + '_prelu')

    return y

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
