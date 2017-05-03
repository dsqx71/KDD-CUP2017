from  config import cfg
import pipeline
import util
import feature
import dataloader

from datetime import datetime, timedelta
from model import model
import tensorflow as tf
import logging
import argparse
import numpy as np
import pandas as pd
import time
import os

def pipeline(args):

    ### Experiment setting
    update_feature = args.update_feature
    resume_epoch = args.resume_epoch
    lr = args.lr

    exp_name = 'RNN'
    num_epoch = 15
    
    save_period = 200
    log_period = 50
    lr_scheduler_period = 5

    lr_decay = 0.5
    batchsize= 256
    num_tf_thread = 8
    clip_grad = 5.0

    ### step1 : prepare data
    data = feature.PreprocessingRawdata(update_feature=update_feature)
    label = feature.GetLabels(data)
    data_standardized = feature.Standardize(data)

    data_train, data_validation, data_test, label_train, label_validation, label_test = \
        feature.SplitData(data_standardized, label)

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
    
    # Get input shapes
    feature_name = data.minor_axis
    shapes= {}
    input_nodes = list(cfg.model.link.keys()) + ['weather'] 
    for key in input_nodes:
        shapes[key] = len([item for item in feature_name if item.startswith(key)])
    # Build Graph
    prediction, loss, metric, label_sym = model.Build(shapes)

    # Optimizer
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Clip gradient
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # gvs = optimizer.compute_gradients(loss)
    # capped_gvs = [(tf.clip_by_value(grad, -clip_grad, clip_grad), var) for grad, var in gvs]
    # optimizer = optimizer.apply_gradients(capped_gvs)
    # create session ans saver
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_tf_thread))
    # saver = tf.train.Saver()
    
    # Build the summary operation and summary writer
    # Training_MAPE = tf.placeholder(shape=[], dtype=tf.float32, name='Training_MAPE')
    # Validation_MAPE = tf.placeholder(shape=[], dtype=tf.float32, name='Validation_MAPE')
    # training_summary = tf.summary.scalar("Training_MAPE", Training_MAPE)
    # validation_summary = tf.summary.scalar("Validation_MAPE", Validation_MAPE)
    # learning_rate_summary = tf.summary.scalar("Learning_rate", learning_rate)
    # summary_writer = tf.summary.FileWriter(exp_dir, sess.graph)

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
        
        error_training = np.zeros((36))
        count_training = np.zeros((36))

        error_validation = np.zeros((36))
        count_validation = np.zeros((36))
        
        tic = time.time()
        # Training
        for batch in training_loader:
            # concat data and label
            data = batch.data
            data.update(batch.label)
            # print (data['loss_scale:0'].shape, data['loss_scale:0'].dtype)
            data['learning_rate:0'] = lr
            data['is_training:0'] = True
            # Feed data into graph
            _, error, label_batch = sess.run([optimizer, metric, label_sym], feed_dict=data)
            mask = (label_batch == label_batch)
            # Update metric
            error_training = error_training + error.sum(0)
            count_training += mask.sum(0)

        toc = time.time()

        # validation
        for batch in validation_loader:
            # concat data and label
            data = batch.data
            data.update(batch.label)
            data['is_training:0'] = False
            # Feed data into graph
            error, label_batch = sess.run([metric, label_sym], feed_dict=data)
            mask = (label_batch == label_batch)
            # Update metric
            error_validation = error_validation + error.sum(0)
            count_validation += mask.sum(0)

        # Speend and Error
        logging.info("Epoch[{}] Speed:{:.2f} samples/sec Training MAPE={:.5f} Validation_MAPE={:.5f}".format(epoch, 
                    training_loader.data_num/(toc-tic), 
                    error_training.sum() / count_training.sum(), 
                    error_validation.sum() / count_validation.sum()))
        print ('training', (error_training / count_training).reshape(6,6))
        print ('validation', (error_validation / count_validation).reshape(6,6))
        
        # Summary
        # if (epoch % log_period == 0):
        #     train_summ, validation_summ, lr_summ = sess.run([training_summary, 
        #                                                      validation_summary, 
        #                                                      learning_rate_summary],
        #                                             feed_dict={'Training_MAPE:0' : error_training.mean(), 
        #                                                        'Validation_MAPE:0' : error_validation.mean(),
        #                                                        'learning_rate:0' : lr})
        #     summary_writer.add_summary(train_summ, epoch)
        #     summary_writer.add_summary(validation_summ, epoch)
        #     summary_writer.add_summary(lr_summ, epoch)

        # Save checkpoint
        # if (epoch % save_period == 0):
        #     logging.info("Saving model of Epoch[{}]...".format(epoch))
        #     saver.save(sess, exp_dir + '/model', global_step=epoch)

        # Learning rate schedule
        if (epoch % lr_scheduler_period == 0):
            lr *= lr_decay

    logging.info("Optimization Finished!")

    # Prediction
    keys = list(prediction.keys())
    keys.sort()
    prediction = [prediction[key] for key in keys]
    traveltime_result = []
    for batch in testing_loader:
        data = batch.data
        data['is_training:0'] = False
        aux = batch.aux
        # Feed data into graph
        pred = sess.run(prediction, feed_dict=data)
        for index, key in enumerate(keys):
            intersection, tollgate = key.split('_')
            tollgate = tollgate[-1]
            time_now = cfg.time.test_timeslots[aux+6:aux+12]
            for i in range(6):
                avg_time = pred[index][0][i]
                left = datetime.strptime(time_now[i], "%Y-%m-%d %H:%M:%S")
                right = left + timedelta(minutes=cfg.time.time_interval)
                
                item = dict(intersection_id=intersection,tollgate_id=tollgate, 
                            time_window='[{},{})'.format(left, right), avg_travel_time=avg_time)
                traveltime_result.append(item)
    
    # save prediction
    traveltime_result = pd.DataFrame(traveltime_result, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'] )
    traveltime_result.to_csv(os.path.join(cfg.data.prediction_dir,'{}_travelTime.csv'.format(exp_name)), 
                            sep=',', header=True,index=False)
    logging.info('Prediction Finished!')
    sess.close()

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_feature', type=int, choices=[0, 1], default=0,
                        help='Indicate whether to create or update hand-craft features')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--resume_epoch', type=int,
                        help='Resume training')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # Start Pipeline
    logging.info("Check Arguments: {}".format(args))
    args.update_feature = bool(args.update_feature)
    pipeline(args)



