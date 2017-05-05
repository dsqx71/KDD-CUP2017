from config import cfg
from .layers import * 
from .metric import * 

def Build(input_size):

    link = cfg.model.link
    route = cfg.model.route
    node_type = cfg.model.node_type

    task1_output = cfg.model.task1_output
    task2_output = cfg.model.task2_output

    # total timestep number : 6 + 6
    timestep_interval = cfg.time.time_interval
    output_interval = 20

    rnn_num_layers = 2
    rnn_dim_hidden = 6

    graph_dim_hidden = 6

    embedding_num = 72
    embedding_feature_num = 2

    encoder_num_timesteps = 120 // timestep_interval
    decoder_num_timesteps = 120 // output_interval

    ### data
    inputs = {}
    encoder_nodes_fw = {}
    decoder_nodes_fw = {}

    states_fw = {}
    output_fw = {}
    
    # aux
    is_training = tf.placeholder(name='is_training', dtype=tf.bool, shape=[])
    loss_scale = tf.placeholder(name='loss_scale', dtype=tf.float32, shape=[None, decoder_num_timesteps])

    ### label
    labels = {}
    label_out = []
    # task1
    keys = list(task1_output.keys())
    keys.sort()
    for key in keys:
        for item in task1_output[key]:
            label_name = '{}_{}'.format(key,item)
            print (label_name)
            labels[label_name] = tf.placeholder(name=label_name, shape=(None, decoder_num_timesteps), dtype=tf.float32)
            label_out.append(labels[label_name])

    # task2 
    keys = list(task2_output.keys())
    keys.sort()
    for key in keys:
        for item in range(task2_output[key]):
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name, shape=(None, decoder_num_timesteps), dtype=tf.float32)
            label_out.append(labels[label_name])
    label_out = tf.concat(label_out, axis=1)
    ### Input data and RNN cell
    time = tf.placeholder(tf.int32, shape=(None), name = 'time')
    weather = tf.placeholder(tf.float32, shape=(None, encoder_num_timesteps + decoder_num_timesteps, input_size['weather']), name='weather')    
    embeddings = {}
    for node in link:
        inputs[node] = tf.placeholder(tf.float32, shape=(None, encoder_num_timesteps, input_size[node]), name = node)
        inputs[node] = tf.cond(is_training, 
                           lambda : inputs[node] + \
                           tf.random_normal(tf.shape(inputs[node]), mean=0, stddev=0.05) * inputs[node] + \
                           tf.random_normal(tf.shape(inputs[node]), mean=0, stddev=0.03),
                           lambda : inputs[node])
        
        # LSTM Cell
        encoder_single_cell_fw = tf.contrib.rnn.GRUCell(rnn_dim_hidden)
        decoder_single_cell_fw = tf.contrib.rnn.GRUCell(rnn_dim_hidden)
        
        encoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([encoder_single_cell_fw] * rnn_num_layers)
        decoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([decoder_single_cell_fw] * rnn_num_layers)

        states_fw[node] = encoder_nodes_fw[node].zero_state(tf.shape(inputs[node])[0], tf.float32)
        shape = tf.stack([tf.shape(inputs[node])[0], rnn_dim_hidden + embedding_feature_num])
        output_fw[node] = [tf.fill(shape, 0.0)]

    # Build Graph
    with tf.variable_scope("Embedding", regularizer=tf.contrib.layers.l2_regularizer(0.3)):
        # Embedding of node
        for node in link:
            embeddings[node] = tf.Variable(tf.random_uniform([embedding_num, embedding_feature_num], -0.5, 0.5), 
                                                              name= node + '_embedding', 
                                                              trainable=True)
    
    
    flag = False
    flags = {value : False for value in node_type.values()}
    with tf.variable_scope("RNN", 
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                    mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32),
                            regularizer=tf.contrib.layers.l2_regularizer(0.05)):

        flags_rnn = {value : False for value in node_type.values()}
        for timestep in range(encoder_num_timesteps):
            time_now = (time + timestep) % embedding_num
            for node in link:
                input_data = inputs[node][:, timestep, :]
                embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)
                embedding.set_shape([None, embedding_feature_num])
                weather_now = weather[:, timestep, :]
                data = tf.concat(axis=1, values=[input_data, embedding, weather_now])  

                # concat output of relevant nodes
                with tf.variable_scope('Graph_Convolution_{}'.format(node)) as scope:
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    link_feature = []
                    for i in range(len(link[node])):
                        link_feature.append(output_fw[link[node][i]][-1])
                    GC = Graph_Convolution(link_feature, out_dim=graph_dim_hidden, name='GC')
        
                data = tf.concat(axis=1, values=[data, GC])
                
                # RNN
                with tf.variable_scope('Encoder-GRU-{}'.format(cfg.model.node_type[node]), 
                                        initializer=tf.orthogonal_initializer(gain=1.0)) as scope:
                    if flags_rnn[cfg.model.node_type[node]] == True:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        flags_rnn[cfg.model.node_type[node]] = True
                    tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                    tmp = tf.concat(axis=1, values=[embedding, tmp])
                    output_fw[node].append(tmp)
                flag = True

        # decoder network
        flag = False
        flags_rnn = {value : False for value in node_type.values()}
        for timestep in range(encoder_num_timesteps, encoder_num_timesteps+decoder_num_timesteps):
            time_now = (time + timestep) % embedding_num
            for node in link:
                input_data = output_fw[node][-1]
                embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)
                embedding.set_shape([None, embedding_feature_num])                      
                weather_now = weather[:, timestep, :]
                data = tf.concat(axis=1, values=[input_data, embedding, weather_now])
                
                # concat output of relevant nodes
                with tf.variable_scope('Graph_Convolution_{}'.format(node)) as scope:
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    link_feature = []
                    for i in range(len(link[node])):
                        link_feature.append(output_fw[link[node][i]][-1])
                    GC = Graph_Convolution(link_feature, out_dim=graph_dim_hidden, name='GC')
                
                data = tf.concat(axis=1, values=[data, GC])
                
                # RNN
                with tf.variable_scope('Decoder-GRU-{}'.format(cfg.model.node_type[node]), 
                                        initializer=tf.orthogonal_initializer(gain=1.0)) as scope:
                    if flags_rnn[cfg.model.node_type[node]] == True:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        flags_rnn[cfg.model.node_type[node]] = True
                    tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                    tmp = tf.concat(axis=1, values=[embedding, tmp])
                    output_fw[node].append(tmp)
                flag = True

        ### Model Output
        prediction = {}
        task1_prediction = {}
        task2_prediction = {}
        with tf.variable_scope("task1_net", regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            for node in task1_output:
                for i in range(2):
                    target = task1_output[node][i]
                    pred_result = []
                    with tf.variable_scope("{}_{}".format(node, target)):
                        for time in range(decoder_num_timesteps):
                            data = []
                            # concat hidden layers of all revelent link
                            for path in route[node][i]:
                                data.append(output_fw[str(path)][1+encoder_num_timesteps+time])
                            data = tf.concat(axis=1, values=data)
                            if time > 0:
                                tf.get_variable_scope().reuse_variables()
                            data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = 1, name='{}_{}_{}'.format(node, target, 'fc1'), with_bn=False)
                            pred_result.append(data)  
                    data = tf.concat(pred_result, axis=1)
                    prediction['{}_{}'.format(node, target)] = data
                    task1_prediction['{}_{}'.format(node, target)] = data

        with tf.variable_scope("task2_net", 
                                initializer=tf.contrib.layers.variance_scaling_initializer(factor=2, 
                                    mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01)):
            
            for key in task2_output:
                num_output = task2_output[key]
                for item in range(num_output):

                    with tf.variable_scope('{}_{}'.format(key, item)):
                        pred_result = []
                        for time in range(decoder_num_timesteps):
                            data = output_fw[key][1+encoder_num_timesteps+time]
                            if time > 0:
                                tf.get_variable_scope().reuse_variables()
                            data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = 1, name='{}_{}_{}'.format(key, item, 'fc1'), with_bn=False)
                            pred_result.append(data)
                        data = tf.concat(pred_result, axis=1)

                    prediction['{}_{}'.format(key, item)] = data
                    task2_prediction['{}_{}'.format(key, item)] = data

        ### Loss and Metric
        with tf.variable_scope('Loss_Metric'):
            
            loss_list = []
            metric_list = []
            keys = list(labels.keys())
            keys.sort()
            for key in keys:
                if key.startswith('tollgate'):
                    tmp = 1
                else:
                    tmp =  1
                loss = L1_loss(data=prediction[key], label=labels[key], scale=loss_scale * tmp)
                metric = L1_loss(data=prediction[key], label=labels[key], scale=1.0)    
                
                loss_list.append(loss)
                metric_list.append(metric)
            
            loss = tf.concat(loss_list, axis=1)
            metric = tf.concat(metric_list, axis=1)

    return task1_prediction, task2_prediction, loss, metric, label_out
