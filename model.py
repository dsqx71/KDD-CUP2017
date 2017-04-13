from bnlstm import BNLSTMCell
from config import cfg
import tensorflow as tf

def FC(x, in_dim, out_dim, name, activation='relu', is_training=True, with_bn=True):
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
                                        fused=True,
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

def ExtractFeature(data, in_dim, out_dim, num_hidden, num_layer, is_training=True):
    
    mask = MultiLayerFC('InputGate', data=data, in_dim=in_dim, out_dim=in_dim, 
                        num_hidden=num_hidden, num_layer=num_layer, activation='sigmoid',is_training=is_training)
    data = data * mask
    data = MultiLayerFC('ExtractFeature', data, in_dim, out_dim, num_hidden, num_layer, is_training=is_training)
    
    return data

def L1_loss(name, data, label, scale=1.0):
    """
     Sparse L1 regression Loss
    """
    condition = tf.sign(label)
    label = tf.where(condition > 0, label, data + 1E-7)
    loss =  tf.abs(data - label, name) / label * scale
    zeros = tf.constant(0, dtype=tf.float32, shape=loss.get_shape())
    loss = tf.where(condition > 0, loss, zeros)
    loss = tf.reduce_sum(loss)
    num_valid = tf.reduce_sum(tf.cast(condition, tf.float32))
    loss = loss / num_valid
    return loss

def GetRNN(batch_size, input_size):

    link = cfg.model.link
    route = cfg.model.route
    node_type = cfg.model.node_type

    task1_output = cfg.model.task1_output
    task2_output = cfg.model.task2_output

    # total timestep number : 24 + 6 = 30
    timestep_interval = cfg.time.time_interval
    output_interval = 20

    rnn_num_layers = 4
    rnn_dim_hidden = 16

    feature_num_layers = 2
    feature_dim_hidden = 32
    feature_dim_output = 16

    recursive_num_layers = 1
    recursive_dim_hidden = 16
    recursive_dim_output = 16

    embedding_num = 72
    embedding_feature_num = 8

    encoder_num_timesteps = 120 // timestep_interval
    decoder_num_timesteps = 120 // output_interval

    ### data
    inputs = {}
    encoder_nodes_fw = {}
    decoder_nodes_fw = {}

    states_fw = {}
    output_fw = {}

    ### label
    labels = {}
    is_training = tf.placeholder(name='is_training', dtype=tf.bool, shape=[])
    # task1
    for key in task1_output:
        for item in task1_output[key]:
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name, shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    # task2 
    for key in task2_output:
        for item in range(task2_output[key]):
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name, shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    ### Input data and RNN cell
    time = tf.placeholder(tf.int32, shape=(batch_size), name = 'time')
    weather = tf.placeholder(tf.float32, shape=(batch_size, encoder_num_timesteps + decoder_num_timesteps, input_size['weather']), name='weather')    
    embeddings = {}
    for node in link:
        # Input, input shape vary from node to node
        inputs[node] = tf.placeholder(tf.float32, shape=(batch_size, encoder_num_timesteps, input_size[node]), name = node)
        inputs[node] = tf.cond(is_training, 
                           lambda : inputs[node] + \
                           tf.random_normal(inputs[node].get_shape(), mean=0, stddev=1E-2) * inputs[node] + \
                           tf.random_normal(inputs[node].get_shape(), mean=0, stddev=1E-3),
                           lambda : inputs[node])

        embeddings[node] = tf.Variable(tf.random_uniform([embedding_num, embedding_feature_num], -1.0, 1.0), name= node + '_embedding', trainable=True)
        
        # LSTM Cell
        encoder_single_cell_fw = tf.contrib.rnn.GRUCell(rnn_dim_hidden)
        decoder_single_cell_fw = tf.contrib.rnn.GRUCell(rnn_dim_hidden)
        
        encoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([encoder_single_cell_fw] * rnn_num_layers)
        decoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([decoder_single_cell_fw] * rnn_num_layers)

        states_fw[node] = encoder_nodes_fw[node].zero_state(batch_size, tf.float32)
        output_fw[node] = [tf.constant(value=0, shape=(batch_size, rnn_dim_hidden), dtype=tf.float32)]

    # Build Graph
    flags = {value : False for value in node_type.values()}
    with tf.variable_scope("RNN", 
                            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=8, 
                                    mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)):
        flag=False
        for timestep in range(encoder_num_timesteps):
            time_now = (time + timestep) % embedding_num
            for node in link:
                input_data = inputs[node][:, timestep, :]
                embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)
                weather_now = weather[:, timestep, :]
                data = tf.concat(axis=1, values=[input_data, embedding, weather_now])  
                
                # concat output of relevant nodes
                link_feature = []
                for i in range(len(link[node])):
                    link_feature.append(output_fw[link[node][i]][-1])
                link_feature = tf.concat(axis=1, values=link_feature)      

                with tf.variable_scope('node_type_{}'.format(node_type[node])) as scope:
                    if flags[node_type[node]] == True:
                        scope.reuse_variables()
                    else:
                        flags[node_type[node]] = True
                    data = ExtractFeature(data=data,
                                          in_dim=data.get_shape()[1], 
                                          out_dim = feature_dim_output, 
                                          num_hidden = feature_dim_hidden, 
                                          num_layer = feature_num_layers,
                                          is_training = is_training)

                with  tf.variable_scope(node) as scope:
                    if (timestep == 0) is False:
                        scope.reuse_variables()
                    
                    link_feature = MultiLayerFC('Graph', 
                                                    data=link_feature, 
                                                    in_dim=link_feature.get_shape()[1], 
                                                    out_dim=recursive_dim_output, 
                                                    num_hidden=recursive_dim_hidden, 
                                                    num_layer=recursive_num_layers, 
                                                    activation='relu', is_training=is_training)

                data = tf.concat(axis=1, values=[data, link_feature])
                
                # RNN
                with tf.variable_scope('Encoder-GRU', initializer=tf.orthogonal_initializer(gain=2.00)) as scope:
                    if flag == True:
                        tf.get_variable_scope().reuse_variables()
                    tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                    output_fw[node].append(tmp)
                flag = True

        # decoder network
        flag = False
        for timestep in range(encoder_num_timesteps, encoder_num_timesteps+decoder_num_timesteps):
            time_now = (time + timestep) % embedding_num
            for node in link:
                input_data = output_fw[node][-1]
                embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)                      
                weather_now = weather[:, timestep, :]
                data = tf.concat(axis=1, values=[input_data, embedding, weather_now])
                
                # concat output of relevant nodes
                link_feature = []
                for i in range(len(link[node])):
                    link_feature.append(output_fw[link[node][i]][-1])
                link_feature = tf.concat(axis=1, values=link_feature)      

                with  tf.variable_scope(node) as scope:
                    scope.reuse_variables()
                    link_feature = MultiLayerFC('Graph', 
                                                data=link_feature, 
                                                in_dim=link_feature.get_shape()[1], 
                                                out_dim=recursive_dim_output, 
                                                num_hidden=recursive_dim_hidden, 
                                                num_layer=recursive_num_layers, 
                                                activation='relu', 
                                                is_training=is_training)

                data = tf.concat(axis=1, values=[data, link_feature])
                
                # RNN
                with tf.variable_scope('Decoder-GRU', initializer=tf.orthogonal_initializer(gain=2.00)) as scope:
                    if flag == True:
                        tf.get_variable_scope().reuse_variables()
                    tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                    output_fw[node].append(tmp)
                flag = True

        ### Model Output
        prediction = {}
        with tf.variable_scope("task1_net"):
            for node in task1_output:
                for i in range(2):
                    target = task1_output[node][i]
                    data = []
                    # concat hidden layers of all revelent link
                    num_path = len(route[node])
                    for path in route[node][i]:
                        data.extend(output_fw[str(path)][1:])
                    data = tf.concat(axis=1, values=data)

                    data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = decoder_num_timesteps, name='{}_{}_{}'.format(node, target, 'fc1'), with_bn=True)
                    data0 = data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = decoder_num_timesteps, name='{}_{}_{}'.format(node, target, 'fc2'), with_bn=False)
                    data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = decoder_num_timesteps, name='{}_{}_{}'.format(node, target, 'fc3'), with_bn=False)
                    data = data + data0
                    
                    prediction['{}_{}'.format(node, target)] = data
        
        with tf.variable_scope("task2_net"):
            for node in task2_output:
                data  = []
                num_output = task2_output[node]
                # concat hidden layers of all revelent link
                for key in link[node]:
                    data.extend(output_fw[key][1:])

                # prediction network
                data = tf.concat(axis=1, values=data)
                data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim= decoder_num_timesteps*num_output, name='{}_{}'.format(node, 'fc1'), with_bn=True)
                data0 = data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim= decoder_num_timesteps*num_output, name='{}_{}'.format(node, 'fc2'), with_bn=False)
                data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim= decoder_num_timesteps*num_output, name='{}_{}'.format(node, 'fc3'), with_bn=False)
                data = data + data0
                data = tf.split(axis=1, num_or_size_splits=num_output, value=data)
                
                for i in range(num_output):
                    prediction['{}_{}'.format(node, i)] = data[i]
        
        ### Loss and Metric
        with tf.variable_scope('Loss'):
            loss = []
            for key in labels:
                cost = L1_loss(data=prediction[key], label=labels[key], scale=1.0, name=key)
                loss.append(cost)
            loss = tf.stack(loss)

    return prediction, loss
