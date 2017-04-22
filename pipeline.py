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
    exp_name = 'RNN-0'
    num_epoch = 600
    update_feature = args.update_feature
    resume_epoch = args.resume_epoch
    lr = args.lr

    save_period = 200
    log_period = 10
    lr_scheduler_period = 10

    lr_decay = 0.85
    batchsize= 256
    num_tf_thread = 8

    ### step1 : prepare data

    # Extracting basic features from rawdata
    volume_feature, trajectory_feature, weather_feature, link_feature = \
            feature.PreprocessingRawdata(update_feature=update_feature)

    # Combine all basic features
    data = feature.CombineBasicFeature(volume_feature, trajectory_feature, weather_feature, link_feature)
    label = feature.GetLabels(data)

    # Filling missing values and convert data
    data = feature.FillingMissingData(data)

    # Suduce mean and div std
    data = feature.Standardize(data)

    # Split data into training data and testing data
    data_train, data_validation, data_test, label_train, label_validation, label_test = \
        feature.SplitData(data, label)

    # Get data iterator
    training_loader = dataloader.DataLoader(data=data_train,
                                            label = label_train, 
                                            batchsize = batchsize,
                                            time= cfg.time.train_timeslots,
                                            mode='train')

    validation_loader = dataloader.DataLoader(data = data_validation,
                                              label = label_validation,
                                              batchsize = 1,
                                              time = cfg.time.validation_timeslots,
                                              mode='validation')

    testing_loader = dataloader.DataLoader(data = data_test,
                                           label = label_test,
                                           batchsize = 1,
                                           time = cfg.time.test_timeslots,
                                           mode='test')
        
    ### step2 : training

    # files
    exp_dir = os.path.join(exp_name)
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)

    # Get Graph
    logging.info('Building Computational Graph...')
    
    shapes = {key:data[key].shape[1] for key in data}
    prediction, loss, metric = model.GetRNN(shapes)

    # Optimizer
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # create session ans saver
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_tf_thread))
    # saver = tf.train.Saver()
    
    # Build the summary operation and summary writer
    Training_MAPE = tf.placeholder(shape=[], dtype=tf.float32, name='Training_MAPE')
    Validation_MAPE = tf.placeholder(shape=[], dtype=tf.float32, name='Validation_MAPE')
    training_summary = tf.summary.scalar("Training_MAPE", Training_MAPE)
    validation_summary = tf.summary.scalar("Validation_MAPE", Validation_MAPE)
    learning_rate_summary = tf.summary.scalar("Learning_rate", learning_rate)
    summary_writer = tf.summary.FileWriter(exp_dir, sess.graph)

    # Model params
    if resume_epoch == 0:
        # initializing
        init = tf.global_variables_initializer()
        logging.info('Initializing params...')
        sess.run(init)
    else:
        ## Loading
        logging.info('Loading the model of epoch[{}]...'.format(resume_epoch))
        saver.restore(sess, exp_dir + '/model-{}'.format(resume_epoch))

    #training
    logging.info("Starting training...")
    for epoch in range(resume_epoch+1, num_epoch+1):
        # Reset loader and metric
        training_loader.reset()
        validation_loader.reset()
        
        error_training = np.zeros((6))
        count_training = 0.0

        error_validation = np.zeros((6))
        count_validation = 0.0
        
        tic = time.time()
        # Training
        for batch in training_loader:
            # concat data and label
            data = batch.data
            data.update(batch.label)
            data['learning_rate:0'] = lr
            data['is_training:0'] = True 
            # Feed data into graph
            _, error = sess.run([optimizer, metric], feed_dict=data)
            
            # Update metric
            error_training = error_training + error
            count_training += 1

        toc = time.time()

        # validation
        for batch in validation_loader:
            # concat data and label
            data = batch.data
            data.update(batch.label)
            data['is_training:0'] = False
            # Feed data into graph
            error = sess.run(metric, feed_dict=data)
            
            # Update metric
            error_validation = error_validation + error
            count_validation += 1

        # Speend and Error 
        error_training = error_training / count_training
        error_validation = error_validation / count_validation

        logging.info("Epoch[{}] Speed:{:.2f} samples/sec Training MAPE={:.5f} Validation_MAPE={:.5f}".format(epoch, 
                    training_loader.data_num/(toc-tic), error_training.mean(), error_validation.mean()))
        
        # Summary
        if (epoch % log_period == 0):
            train_summ, validation_summ, lr_summ = sess.run([training_summary, 
                                                             validation_summary, 
                                                             learning_rate_summary],
                                                    feed_dict={'Training_MAPE:0' : error_training.mean(), 
                                                               'Validation_MAPE:0' : error_validation.mean(),
                                                               'learning_rate:0' : lr})
            summary_writer.add_summary(train_summ, epoch)
            summary_writer.add_summary(validation_summ, epoch)
            summary_writer.add_summary(lr_summ, epoch)

        # Save checkpoint
        if (epoch % save_period == 0):
            logging.info("Saving model of Epoch[{}]...".format(epoch))
            saver.save(sess, exp_dir + '/model', global_step=epoch)

        # Learning rate schedule
        if (epoch % lr_scheduler_period == 0):
            lr *= lr_decay

    logging.info("Optimization Finished!")
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



