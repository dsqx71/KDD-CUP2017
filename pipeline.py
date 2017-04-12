from  config import cfg
import pipeline
import util
import feature
import model
import dataloader

import time
import tensorflow as tf
import logging
import argparse
import numpy as np
import pandas as pd
import os

def pipeline(args):

    ### Experiment setting
    exp_name = 'RNN_Exp1'
    num_epoch = 1000
    force_update = args.update_feature
    resume_epoch = args.resume_epoch
    lr = args.lr

    save_interval = 20
    lr_decay = 0.90
    batchsize= 128
    num_tf_thread = 8

    ### step1 : prepare data

    # Extracting basic features from rawdata
    volume_feature, trajectory_feature, weather_feature, link_feature = \
            feature.PreprocessingRawdata(force_update=force_update)

    # Combine all basic features
    data = feature.CombineBasicFeature(volume_feature, trajectory_feature, weather_feature, link_feature)
    label = feature.GetLabels(data)

    # Suduce mean and div std
    data = feature.Standardize(data)

    # Filling missing values and convert data
    data = feature.FillingMissingData(data)

    # Split data into training data and testing data
    data_train, data_validation, data_test, label_train, label_validation, label_test = \
        feature.SplitData(data, label)

    # Get data iterator
    loader = dataloader.DataLoader(data=data_train,
                        label = label_train, 
                        batchsize = batchsize,
                        time= cfg.time.train_timeslots,
                        is_train=True)
        
    ### step2 : training

    # files 
    exp_dir = os.path.join(cfg.data.checkpoint_dir, exp_name)
    model_dir = exp_dir + "/model"
    summary_dir = exp_dir + "/summary"

    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    if os.path.exists(summary_dir) is False:
        os.mkdir(summary_dir)

    # Graph and params
    if resume_epoch == -1:
        # Get Computing Graph
        logging.info('Building Computational Graph...')
        shapes = {key:data[key].shape[1] for key in data}
        prediction, loss = model.GetLSTM(batchsize, shapes)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Optimizer
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        tf.add_to_collection('optimizer', optimizer)
        tf.add_to_collection('loss', loss)
        for key in prediction:
            tf.add_to_collection(key, prediction[key])

        # Create session
        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_tf_thread))
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        logging.info('Initializing params...')
        sess.run(init)
        
        # Save Graph and params
        saver.save(sess, model_dir + '/model', global_step=0)
    else:
        logging.info('Loading the model of epoch[{}]...'.format(resume_epoch))
        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_tf_thread))
        saver = tf.train.import_meta_graph(model_dir +'/model-{}.meta'.format(resume_epoch))
        saver.restore(sess, model_dir + '/model-{}'.format(resume_epoch))
        optimizer = tf.get_collection('optimizer')[0]
        loss = tf.get_collection('loss')[0]

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    #training
    logging.info("Starting training...")
    for epoch in range(resume_epoch+1, num_epoch):
        # reset loader and metric
        loader.reset()
        tic = time.time()
        error_all = np.zeros((11))
        count = 0.0
        
        for batch in loader:
            
            # concat data and label
            data = batch.data
            data.update(batch.label)
            data['learning_rate:0'] = lr
            
            # Feed data into graph
            _, error = sess.run([optimizer, loss], feed_dict=data)
            
            # Update metric
            error_all = error_all + error
            count += 1
        
        # Speend and Error 
        error_all = error_all / count
        toc = time.time()
        logging.info("Epoch[{}] Speed:{:.2f} samples/sec Overal loss={:.5f}".format(epoch, loader.data_num/(toc-tic), error_all.mean()))
        
        # save 
        if (epoch % save_interval == 0) and (epoch !=0) :
            lr *= lr_decay
            logging.info("Saving model of Epoch[{}]...".format(epoch))
            saver.save(sess, model_dir + '/model', global_step=epoch)

    loging.info("Optimization Finished!")
    sess.close()
    
    

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_feature', type=int, choices=[0, 1], default=0,
                        help='indicate whether to create or update hand-craft features')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--resume_epoch', type=int,
                        help='resume training')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # Start Pipeline
    logging.info("Check Arguments: {}".format(args))
    args.update_feature = bool(args.update_feature)
    pipeline(args)



