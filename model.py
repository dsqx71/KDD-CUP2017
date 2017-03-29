from tensorflow.python.ops import array_ops

import tensorflow as tf
import config.cfg.model.link as link
import config.cfg.model.task1_output as task1_output
import config.cfg.model.task2_output as task2_output
import config.cfg.model.route as route

def FC(x, in_dim, out_dim, name):
    """
    Fully connect + relu
    """
    W = tf.get_variable(name=name+'_weight', shape=[in_dim, out_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    b = tf.get_variable(name=name+'_bias', shape=[out_dim], initializer=tf.zeros_initializer)
    y = tf.nn.relu(tf.matmul(x, W) + b, name=name+'_relu')
    return y

def spatial_network(cell, hidden, in_dim, out_dim, num_hidden=8):

    data = array_ops.concat(1, [cell, hidden])
    fc1 = FC(data, in_dim=in_dim*2, out_dim=num_hidden, name='fc1')
    fc2 = FC(fc1, in_dim=num_hidden, out_dim=out_dim*2, name='fc2')
    cell, hidden = array_ops.split(split_dim=1, num_split=2, value=fc2)
    state = (cell, hidden)
    return state

def L1_loss(name, data, label, scale=1.0):

    loss = tf.abs(data - label, name) / label * scale

    return loss

def LSTM():
    # total timestep number : 24 + 6 = 30
    timestep_interval = 5
    output_interval = 20

    batch_size = 10
    num_layers = 5
    dim_hidden = 16
    feature_size = 5

    assert 120 % timestep_interval == 0, 'wrong timestep interval'

    encoder_num_timesteps = 120 // timestep_interval
    decoder_num_timesteps = 120 // output_interval

    # data
    inputs = {}

    encoder_nodes_fw = {}
    # encoder_nodes_bw = {}
    decoder_nodes_fw = {}
    # decoder_nodes_bw = {}

    states_fw = {}
    states_bw = {}
    output_fw = {}
    output_bw = {}
    output = {}

    # label
    labels = {}

    # task1
    for key in task1_output:
        for item in range(task1_output[key]):
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name,shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    # task2
    for key in task2_output:
        for item in task2_output[key]:
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name,shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    # Nodes of network
    for node in link:
        # Input
        inputs[node] = tf.placeholder(tf.float32, shape=(None, encoder_num_timesteps, feature_size), name=node+'input')

        # LSTM Cell
        encoder_single_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        # encoder_single_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        decoder_single_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        # decoder_single_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        # encoder : forward
        encoder_nodes_fw[node] = tf.nn.rnn_cell.MultiRNNCell([encoder_single_cell_fw] * num_layers)

        # decoder : forward and backward
        decoder_nodes_fw[node] = tf.nn.rnn_cell.MultiRNNCell([decoder_single_cell_fw] * num_layers)
        # decoder_nodes_bw[node] = tf.nn.rnn_cell.MultiRNNCell([decoder_single_cell_bw] * num_layers)

        # states of forward and backward
        states_fw[node] = encoder_nodes_fw[node].zero_state(batch_size, tf.float32)
        # states_bw[node] = decoder_nodes_bw[node].zero_state(batch_size, tf.float32)

        link[node].append(node)
        output[node] = []

    # Build Graph
    with tf.variable_scope("Tree-StructureRNN"):
        with tf.variable_scope("Encoder"):
            with tf.variable_scope("Forward"):
                flag=False
                for timestep in range(encoder_num_timesteps):
                    for node in link:
                        # input feature
                        data = inputs[node][:, timestep, :]
                        # concat all cells and hidden layers of in_link
                        num_intop = len(link[node])
                        new_cell  = [[] for i in range(num_layers)]
                        new_hidden = [[] for i in range(num_layers)]
                        for i in range(num_intop):
                            link_name = link[node][i]
                            states = states_fw[link_name]
                            for j in range(num_layers):
                                new_cell[j].append(states[j].c)
                                new_hidden[j].append(states[j].h)

                        for j in range(num_layers):
                            new_cell[j] = array_ops.concat(concat_dim=1,values=new_cell[j])
                            new_hidden[j] = array_ops.concat(concat_dim=1,values=new_hidden[j])

                        # Tree-structure Network
                        with  tf.variable_scope(node) as scope:
                            # this network model spatial information
                            in_dim = num_intop * dim_hidden
                            states = []
                            for i in range(num_layers):
                                if ((timestep == 0) and (i == 0)) is False:
                                    scope.reuse_variables()
                                tmp = spatial_network(cell=new_cell[i], hidden=new_hidden[i], in_dim=in_dim, out_dim= dim_hidden)
                                states.append(tmp)

                        # Recurrent Network
                        with tf.variable_scope('LSTM', initializer=tf.orthogonal_initializer(gain=1.00)) as scope:
                            if flag == True:
                                tf.get_variable_scope().reuse_variables()
                            output_fw[node], states_fw[node] = encoder_nodes_fw[node](data, states)
                        flag = True

        # decoder network
        with tf.variable_scope("Decoder"):
            with tf.variable_scope("forward"):
                flag = False
                for timestep in range(encoder_num_timesteps, encoder_num_timesteps+decoder_num_timesteps):
                    for node in link:
                        data = output_fw[node]
                        # concat in_link
                        num_intop = len(link[node])
                        new_cell  = [[] for i in range(num_layers)]
                        new_hidden = [[] for i in range(num_layers)]
                        for i in range(num_intop):
                            link_name = link[node][i]
                            states = states_fw[link_name]
                            for j in range(num_layers):
                                new_cell[j].append(states[j].c)
                                new_hidden[j].append(states[j].h)

                        for j in range(num_layers):
                            new_cell[j] = array_ops.concat(concat_dim=1,values=new_cell[j])
                            new_hidden[j] = array_ops.concat(concat_dim=1,values=new_hidden[j])

                        # Tree-structure Network
                        with  tf.variable_scope(node) as scope:
                            # this network model spatial information
                            in_dim = num_intop * dim_hidden
                            states = []
                            for i in range(num_layers):
                                if ((timestep == encoder_num_timesteps) and (i == 0)) is False:
                                    scope.reuse_variables()
                                tmp = spatial_network(cell=new_cell[i], hidden=new_hidden[i], in_dim=in_dim, out_dim= dim_hidden)
                                states.append(tmp)

                        # Recurrent Network
                        with tf.variable_scope('LSTM', initializer=tf.orthogonal_initializer(gain=1.00)) as scope:
                            if flag == True:
                                tf.get_variable_scope().reuse_variables()
                            output_fw[node], states_fw[node] = decoder_nodes_fw[node](data, states)
                            output[node].append(output_fw[node])
                        flag = True

        # Output
        prediction = {}
        with tf.variable_scope("task1_net"):
            for node in task1_output:
                data  = []
                num_output = task1_output[node]

                # concat hidden layers of all revelent link
                for key in link[node]:
                    data.extend(output[key])

                # prediction network
                data = array_ops.concat(concat_dim=1, values=data)
                data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim= decoder_num_timesteps*num_output, name='{}_{}'.format(node, 'fc1'))
                data = array_ops.split(split_dim=1, num_split=num_output, value=data)
                for i in range(num_output):
                    prediction['{}_{}'.format(node, i)] = data[i]

        with tf.variable_scope("task1_net"):
            for node in task2_output:
                for i in range(2):
                    target = task2_output[node][i]

                    data = []
                    # concat hidden layers of all revelent link
                    num_path = len(route[node])
                    for path in route[node][i]:
                        data.extend(output[str(path)])
                    data = array_ops.concat(concat_dim=1, values=data)
                    data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = decoder_num_timesteps, name='{}_{}_{}'.format(node, target, 'fc1'))
                    prediction['{}_{}'.format(node, target)] = data

        with tf.variable_scope('loss'):
            loss = []
            for key in labels:
                cost = L1_loss(data=prediction[key], label=labels[key], scale=1.0, name=key)
                tf.summary.scalar('loss_{}'.format(key), cost)
                loss.append(cost)
            loss = array_ops.concat(concat_dim=1, values=loss, name='concat_all_loss')

    # optimizer
    lr = 1e-4
    adam = tf.train.AdamOptimizer(learning_rate=lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = adam.minimize(loss, global_step=global_step)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # all parameters
    args = tf.get_collection(tf.GraphKeys.VARIABLES, scope="Tree-StructureRNN")
    tot = 0
    for item in args:
        shapes = item.get_shape()
        now = shapes[0].value
        for i in shapes[1:]:
            now*=i.value
        tot += now
    print (tot)