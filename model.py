from tensorflow.python.ops import array_ops
from config import cfg
import tensorflow as tf

def FC(x, in_dim, out_dim, name):
    """
    Fully connect + relu
    """
    W = tf.get_variable(name=name+'_weight', shape=[in_dim, out_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
    b = tf.get_variable(name=name+'_bias', shape=[out_dim], initializer=tf.zeros_initializer)
    y = tf.nn.relu(tf.matmul(x, W) + b, name=name + '_relu')
    return y

def MultiLayerFC(name, data, in_dim, out_dim, num_hidden, num_layer):
    
    if num_layer > 1:
        data = FC(data, in_dim=in_dim, out_dim=num_hidden, name='{}_fc0'.format(name))
        for i in range(1, num_layer-1):
             data= FC(data, in_dim = num_hidden, out_dim = num_hidden, name = '{}_fc{}'.format(name, i))
        data = FC(data, in_dim=num_hidden, out_dim=out_dim, name='{}_fc{}'.format(name, num_layer-1))
    else:
        data = FC(data, in_dim=in_dim, out_dim=out_dim, name='{}_fc0'.format(name))
    
    return data
    
def RecursiveNetwork(cell, hidden, in_dim, out_dim, num_hidden, num_layer):

    cell = MultiLayerFC('RecursiveNetwork_cell', cell, in_dim, out_dim, num_hidden, num_layer)
    hidden = MultiLayerFC('RecursiveNetwork_hidden', hidden, in_dim, out_dim, num_hidden, num_layer)
    state = (cell, hidden)
    
    return state

def ExtractFeature(data, in_dim, out_dim, num_hidden, num_layer):

    mask = MultiLayerFC('InputGate', data, in_dim, in_dim, in_dim, 1)
    data = data * mask
    data = MultiLayerFC('ExtractFeature', data, in_dim, out_dim, num_hidden, num_layer)
    return data

def L1_loss(name, data, label, scale=1.0):
    loss = tf.abs(data - label, name) / label * scale
    return loss

def Network():

    link = cfg.model.link
    route = cfg.model.route
    task1_output = cfg.model.task1_output
    task2_output = cfg.model.task2_output

    # total timestep number : 24 + 6 = 30
    timestep_interval = cfg.time.time_interval
    output_interval = 20

    input_size = 16
    batch_size = 128

    rnn_num_layers = 5
    rnn_dim_hidden = 8

    feature_num_layers= 3
    feature_dim_hidden = 8
    feature_dim_output = 8

    recursive_num_layers = 1
    recursive_dim_hidden = 8

    embedding_num = 72
    embedding_feature_num = 8

    encoder_num_timesteps = 120 // timestep_interval
    decoder_num_timesteps = 120 // output_interval

    ### data
    inputs = {}

    encoder_nodes_fw = {}
    decoder_nodes_fw = {}

    states_fw = {}
    states_bw = {}
    output_fw = {}
    output_bw = {}
    output = {}

    ### label
    labels = {}

    # task1
    for key in task1_output:
        for item in task1_output[key]:
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name,shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    # task2 
    for key in task2_output:
        for item in range(task2_output[key]):
            label_name = '{}_{}'.format(key,item)
            labels[label_name] = tf.placeholder(name=label_name,shape=(batch_size, decoder_num_timesteps), dtype=tf.float32)

    ### Input data and RNN cell
    time = tf.placeholder(tf.int32, shape=(None), name = 'time')    

    for node in link:
        # Input, input shape vary from node to node
        data = tf.placeholder(tf.float32, shape=(None, encoder_num_timesteps, input_size), name = node + 'input')
        embeddings = tf.Variable(tf.random_uniform([embedding_num, rnn_dim_hidden], -1.0, 1.0), name= node + 'embedding')
        embed = tf.nn.embedding_lookup(embeddings, time)
        inputs[node] = array_ops.concat(concat_dim=1, values=[embed, data])
        
        # LSTM Cell
        encoder_single_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim_hidden, state_is_tuple=True)
        decoder_single_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim_hidden, state_is_tuple=True)
        
        encoder_nodes_fw[node] = tf.nn.rnn_cell.MultiRNNCell([encoder_single_cell_fw] * rnn_num_layers)
        decoder_nodes_fw[node] = tf.nn.rnn_cell.MultiRNNCell([decoder_single_cell_fw] * rnn_num_layers)

        states_fw[node] = encoder_nodes_fw[node].zero_state(batch_size, tf.float32)
        
        link[node].append(node)
        output[node] = []

    # Build Graph
    with tf.variable_scope("RNN"):
        with tf.variable_scope("Encoder"):
            with tf.variable_scope("Forward"):
                flag=False
                for timestep in range(encoder_num_timesteps):
                    for node in link:
                        data = inputs[node][:, timestep, :]
                        
                        # concat all cells and hidden layers of relevant nodes
                        num_intop = len(link[node])
                        new_cell  = [[] for i in range(rnn_num_layers)]
                        new_hidden = [[] for i in range(rnn_num_layers)]
                        for i in range(num_intop):
                            link_name = link[node][i]
                            states = states_fw[link_name]
                            for j in range(rnn_num_layers):
                                new_cell[j].append(states[j].c)
                                new_hidden[j].append(states[j].h)

                        for j in range(rnn_num_layers):
                            new_cell[j] = array_ops.concat(concat_dim=1,values=new_cell[j])
                            new_hidden[j] = array_ops.concat(concat_dim=1,values=new_hidden[j])
                        
                        with  tf.variable_scope(node) as scope:
                            if  (timestep > 0):
                                scope.reuse_variables()
                            data = ExtractFeature(data=data,
                                                  in_dim=data.get_shape()[1], 
                                                  out_dim = feature_dim_output, 
                                                  num_hidden = feature_dim_hidden, 
                                                  num_layer = feature_num_layers)
                                
                        with  tf.variable_scope(node) as scope:
                            in_dim = num_intop * rnn_dim_hidden
                            states = []
                            for i in range(rnn_num_layers):
                                if ((timestep == 0) and (i == 0)) is False:
                                    scope.reuse_variables()
                                tmp = RecursiveNetwork(cell = new_cell[i], 
                                                       hidden = new_hidden[i], 
                                                       in_dim = in_dim, 
                                                       out_dim = rnn_dim_hidden,
                                                       num_hidden = recursive_dim_hidden,
                                                       num_layer = recursive_num_layers)
                                states.append(tmp)

                        # RNN
                        with tf.variable_scope('LSTM', initializer=tf.orthogonal_initializer(gain=1.00)) as scope:
                            if flag == True:
                                tf.get_variable_scope().reuse_variables()
                            output_fw[node], states_fw[node] = encoder_nodes_fw[node](data, states)
                        flag = True

        # decoder network
        with tf.variable_scope("Decoder"):
            with tf.variable_scope("Forward"):
                flag = False
                for timestep in range(encoder_num_timesteps, encoder_num_timesteps+decoder_num_timesteps):
                    for node in link:
                        data = output_fw[node]

                        # concat in_link
                        num_intop = len(link[node])
                        new_cell  = [[] for i in range(rnn_num_layers)]
                        new_hidden = [[] for i in range(rnn_num_layers)]
                        for i in range(num_intop):
                            link_name = link[node][i]
                            states = states_fw[link_name]
                            for j in range(rnn_num_layers):
                                new_cell[j].append(states[j].c)
                                new_hidden[j].append(states[j].h)

                        for j in range(rnn_num_layers):
                            new_cell[j] = array_ops.concat(concat_dim=1,values=new_cell[j])
                            new_hidden[j] = array_ops.concat(concat_dim=1,values=new_hidden[j])

                        with  tf.variable_scope(node) as scope:
                            in_dim = num_intop * rnn_dim_hidden
                            states = []
                            for i in range(rnn_num_layers):
                                if ((timestep == encoder_num_timesteps) and (i == 0)) is False:
                                    scope.reuse_variables()
                                tmp = RecursiveNetwork(cell=new_cell[i], 
                                                       hidden=new_hidden[i], 
                                                       in_dim=in_dim, 
                                                       out_dim= rnn_dim_hidden,
                                                       num_hidden = recursive_dim_hidden,
                                                       num_layer = recursive_num_layers)
                                states.append(tmp)

                        with tf.variable_scope('LSTM', initializer=tf.orthogonal_initializer(gain=1.00)) as scope:
                            if flag == True:
                                tf.get_variable_scope().reuse_variables()
                            output_fw[node], states_fw[node] = decoder_nodes_fw[node](data, states)
                            output[node].append(output_fw[node])
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
                        data.extend(output[str(path)])
                    data = array_ops.concat(concat_dim=1, values=data)
                    data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim = decoder_num_timesteps, name='{}_{}_{}'.format(node, target, 'fc1'))
                    prediction['{}_{}'.format(node, target)] = data
        
        with tf.variable_scope("task2_net"):
            for node in task2_output:
                data  = []
                num_output = task2_output[node]

                # concat hidden layers of all revelent link
                for key in link[node]:
                    data.extend(output[key])

                # prediction network
                data = array_ops.concat(concat_dim=1, values=data)
                data = FC(x=data, in_dim=data.get_shape()[1].value, out_dim= decoder_num_timesteps*num_output, name='{}_{}'.format(node, 'fc1'))
                data = array_ops.split(split_dim=1, num_split=num_output, value=data)
                for i in range(num_output):
                    prediction['{}_{}'.format(node, i)] = data[i]
        
        ### Loss and Metric
        with tf.variable_scope('Loss'):
            loss = []
            for key in labels:
                cost = L1_loss(data=prediction[key], label=labels[key], scale=1.0, name=key)
                tf.summary.scalar('Loss_{}'.format(key), cost)
                loss.append(cost)
            loss = array_ops.concat(concat_dim=1, values=loss, name='concat_all_loss')

    return prediction, loss, summary