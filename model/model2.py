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

    window_num = 2

    graph_dim_hidden = 5

    embedding_num = 72
    embedding_feature_num = 5

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
    loss_scale  = tf.placeholder(name='loss_scale',  dtype=tf.float32, shape=[None, decoder_num_timesteps])

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
            labels[label_name] = tf.cond(is_training,
                           lambda : labels[label_name] + \
                           tf.random_normal(tf.shape(labels[label_name]), mean=0, stddev=0.01) * labels[label_name] + \
                           tf.random_normal(tf.shape(labels[label_name]), mean=0, stddev=0.05),
                           lambda : labels[label_name])
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
    embedding_output = {}
    for node in link:
        inputs[node] = tf.placeholder(tf.float32, shape=(None, encoder_num_timesteps, input_size[node]), name = node)
        inputs[node] = tf.cond(is_training,
                           lambda : inputs[node] + \
                           tf.random_normal(tf.shape(inputs[node]), mean=0, stddev=0.05) * inputs[node] + \
                           tf.random_normal(tf.shape(inputs[node]), mean=0, stddev=0.01),
                           lambda : inputs[node])

        # LSTM Cell
        encoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(4), tf.contrib.rnn.GRUCell(4), tf.contrib.rnn.GRUCell(2)])
        decoder_nodes_fw[node] = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(4), tf.contrib.rnn.GRUCell(4), tf.contrib.rnn.GRUCell(2)])

        states_fw[node] = encoder_nodes_fw[node].zero_state(tf.shape(inputs[node])[0], tf.float32)
        shape = tf.stack([tf.shape(inputs[node])[0], 2])
        output_fw[node] = [tf.fill(shape, 0.0)]
        link[node].append(node)

    # Build Graph
    with tf.variable_scope("Embedding",
                           regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
        # Embedding of node
        for node in link:
            embeddings[node] = tf.Variable(tf.random_uniform([embedding_num, embedding_feature_num], -0.03, 0.03),
                                                              name= node + '_Embedding',
                                                              trainable=True)

    flag = False
    flags = {value : False for value in node_type.values()}
    with tf.variable_scope("RNN",
                           initializer=tf.orthogonal_initializer(gain=1.41)
                           regularizer=tf.contrib.layers.l2_regularizer(0.001)):

        flags_rnn = {value : False for value in node_type.values()}
        for timestep in range(encoder_num_timesteps):
            time_now = (time + timestep) % embedding_num
            for window in range(window_num):
                with tf.variable_scope('Graph-Convolution') as scope:
                    if not (timestep == 0 and window == 0):
                        tf.get_variable_scope().reuse_variables()
                    tmp = {}
                    for node in link:
                        tmp[node] = output_fw[node][-1]
                    gc1 = GC(tmp, out_dim=graph_dim_hidden, name='GC1')
                    gc2 = GC(gc1, out_dim=graph_dim_hidden, name='GC2')
                    gc = {}
                    for node in link:
                        gc[node] = [gc2[node]]
                        gc[node] = tf.concat(values=gc[node], axis=1)

                for node in link:
                    input_data = inputs[node][:, timestep, :]
                    embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)
                    embedding.set_shape([None, embedding_feature_num])
                    weather_now = weather[:, timestep, :]
                    data = tf.concat(axis=1, values=[input_data, embedding, weather_now, gc[node]])

                    # RNN
                    with tf.variable_scope('Encoder-GRU-{}'.format(cfg.model.node_type[node]),
                                            initializer=tf.orthogonal_initializer(gain=1.41)) as scope:
                        if flags_rnn[cfg.model.node_type[node]] == True:
                            tf.get_variable_scope().reuse_variables()
                        else:
                            flags_rnn[cfg.model.node_type[node]] = True
                        tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                        output_fw[node].append(tmp)
                    flag = True

        # decoder network
        flag = False
        flags_rnn = {value : False for value in node_type.values()}
        for timestep in range(encoder_num_timesteps, encoder_num_timesteps+decoder_num_timesteps):

            time_now = (time + timestep) % embedding_num
            for window in range(window_num):
                with tf.variable_scope('Graph-Convolution') as scope:
                    if not (timestep == 0 and window == 0):
                        tf.get_variable_scope().reuse_variables()
                    tmp = {}
                    for node in link:
                        tmp[node] = output_fw[node][-1]
                    gc1 = GC(tmp, out_dim=graph_dim_hidden, name='GC1')
                    gc2 = GC(gc1, out_dim=graph_dim_hidden, name='GC2')
                    gc = {}
                    for node in link:
                        gc[node] = [gc2[node]]
                        gc[node] = tf.concat(values=gc[node], axis=1)

                for node in link:
                    input_data = output_fw[node][-1]
                    embedding  = tf.nn.embedding_lookup(embeddings[node], time_now)
                    embedding.set_shape([None, embedding_feature_num])
                    weather_now = weather[:, timestep, :]
                    data = tf.concat(axis=1, values=[input_data, embedding, weather_now, gc[node]])
                    # RNN
                    with tf.variable_scope('Decoder-GRU-{}'.format(cfg.model.node_type[node]),
                                           initializer=tf.orthogonal_initializer(gain=1.41)) as scope:
                        if flags_rnn[cfg.model.node_type[node]] == True:
                            tf.get_variable_scope().reuse_variables()
                        else:
                            flags_rnn[cfg.model.node_type[node]] = True
                        tmp, states_fw[node] = encoder_nodes_fw[node](data, states_fw[node])
                        output_fw[node].append(tmp)
                    flag = True

        ### Model Output
        prediction = {}
        task1_prediction = {}
        task2_prediction = {}
        with tf.variable_scope("task1_net", regularizer=tf.contrib.layers.l1_regularizer(0.01)):
            for node in task1_output:
                for i in range(2):
                    target = task1_output[node][i]
                    pred_result = []
                    with tf.variable_scope("{}_{}".format(node, target)):
                        for timestep in range(decoder_num_timesteps):
                            data = output_fw[node][1+encoder_num_timesteps*window_num+(timestep)*window_num:1+encoder_num_timesteps*window_num+(timestep+1)*window_num]
                            for path in route[node][i]:
                                data.extend(output_fw[str(path)][1+encoder_num_timesteps*window_num+(timestep)*window_num:1+encoder_num_timesteps*window_num+(timestep+1)*window_num])
                            data = tf.concat(axis=1, values=data)
                            if timestep > 0:
                                tf.get_variable_scope().reuse_variables()
                            data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = 1, name='{}_{}_{}'.format(node, target, 'fc1'), with_bn=False)
                            pred_result.append(data)
                    data = tf.concat(pred_result, axis=1) * 10
                    prediction['{}_{}'.format(node, target)] = data
                    task1_prediction['{}_{}'.format(node, target)] = data

        with tf.variable_scope("task2_net",
                                regularizer=tf.contrib.layers.l2_regularizer(0.01)):

            for key in task2_output:
                num_output = task2_output[key]
                for item in range(num_output):

                    with tf.variable_scope('{}_{}'.format(key, item)):
                        pred_result = []
                        for timestep in range(decoder_num_timesteps):
                            time_now = (time + timestep + encoder_num_timesteps) % embedding_num
                            print (time_now)
                            data = output_fw[key][1+encoder_num_timesteps*window_num+(timestep-1)*window_num:1+encoder_num_timesteps*window_num+(timestep+1)*window_num]
                            data = tf.concat(values=data, axis=1)
                            if timestep > 0:
                                tf.get_variable_scope().reuse_variables()
                            data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = 1, name='{}_{}_{}'.format(key, item, 'fc1'), with_bn=False)
                            pred_result.append(data)
                        data = tf.concat(pred_result, axis=1) * 10
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
                    tmp = 0.1
                else:
                    tmp = 1
                loss = L1_loss(data=prediction[key], label=labels[key], scale=loss_scale * tmp)
                metric = L1_loss(data=prediction[key], label=labels[key], scale=1.0)

                loss_list.append(loss)
                metric_list.append(metric)

            loss = tf.concat(loss_list, axis=1)
            metric = tf.concat(metric_list, axis=1)

    return task1_prediction, task2_prediction, loss, metric, label_out
