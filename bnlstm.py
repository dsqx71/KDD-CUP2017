import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class BNLSTMCell(RNNCell):
    """
    Recurrent Batch Normalization
    """
    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

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
                                                is_training=self.training,
                                                scope='xh')

            bn_hh = tf.contrib.layers.batch_norm(hh, 
                                                decay=0.99,
                                                epsilon=1E-3,
                                                center=True,
                                                scale=True,
                                                is_training=self.training,
                                                scope='hh')

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = tf.contrib.layers.batch_norm(new_c,
                                                decay=0.99,
                                                epsilon=1E-3,
                                                center=True,
                                                scale=True,
                                                is_training=self.training,
                                                scope='c')

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)