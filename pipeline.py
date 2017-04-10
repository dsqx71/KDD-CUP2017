from  config import cfg
import pipeline
import util
import feature
import model
import dataloader

import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import os

def pipeline(args):
    ### step1 : prepare data

    # Extracting basic features from rawdata
    volume_feature, trajectory_feature, weather_feature, link_feature = \
        feature.PreprocessingRawdata(force_update=False)

    # Combine all basic features
    data = feature.CombineBasicFeature(volume_feature, trajectory_feature, weather_feature, link_feature)

    # Split data into training data and testing data
    data_train, data_test = feature.SplitData(data)

    # Get labels of training dataset
    labels_train = feature.GetLabels(data_train)

    # Filling missing values and convert data to numpy array
    data_train = feature.FillingMissingData(data_train)
    data_test = feature.FillingMissingData(data_test)

    # Data Iterator
    loader = dataloader.DataLoader(data=data_train,
                    label = labels_train, 
                    batchsize = 128, 
                    time= cfg.time.train_timeslots,
                    is_train=True)
    
    ### step2 : training

    # Get Computing Graph
    logging.info("Building Computing Graph...")
    shapes = {key:data[key].shape[1] for key in data}
    prediction, loss = model.GetLSTM(shapes)

    # Experiment setting
    exp_name = 'RNN_Exp1'
    lr = 0.1
    num_epoch = 500

    model_file = os.path.join(cfg.data.checkpoint_dir, exp_name + '_model')
    summary_file = os.path.join(cfg.data.checkpoint_dir, exp_name + '_summary')

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # optimizer
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # Create the op for initializing variables.
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=0))
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(summary_file, sess.graph)

    logging.info("Start training...")
    for epoch in range(num_epoch):
    
        loader.reset()
        tic = time.time()
        
        for batch in loader:
            data = batch.data
            data.update(batch.label)
            data['learning_rate:0'] = lr
            _, error = sess.run([optimizer, loss], feed_dict=data)
        
        toc = time.time()
        logging.info("Epoch[{}] Speed:{:.2f} samples/sec Overal loss={:.5f}".format(epoch, loader.data_num/(toc-tic), error.tolist()))
        
        if epoch % 10 == 0:
            logging.info("Saving model of Epoch[{}]...".format(epoch))
            saver.save(sess, model_file, global_step=global_step)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_feature', type=int, choices=[0, 1], default=0,
                        help='indicate whether to create or update hand-craft features')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--num_epoch', type=int,
                        help='number of epoches')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # Start Pipeline
    logging.info("Check Arguments: {}".format(args))
    args.update_feature = bool(args.update_feature)
    pipeline(args)



