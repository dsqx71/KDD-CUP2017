import feature
import argparse
import logging

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
    
    ### step2 : training

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_feature', type=int, choices=[0, 1], default=0,
                        help='indicate whether to create or update hand-craft features')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # Start Pipeline
    logging.info("Check Arguments: {}".format(args))
    args.update_feature = bool(args.update_feature)
    pipeline(args)



