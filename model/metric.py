import tensorflow as tf

def L1_loss(data, label, scale=1.0, type='loss'):
    """
     Sparse L1 regression Loss
    """
    condition = tf.sign(label)
    label = tf.where(condition > 0, label, data + 1E-8)
    loss =  tf.abs(data - label, ) / label * scale
    zeros = tf.fill(tf.shape(loss), 0.0)
    loss = tf.where(condition > 0, loss, zeros)
    # if type == 'loss':
    #     loss = tf.where(loss > 0.14, loss, zeros)
    return loss
