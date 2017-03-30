import feature
import argparse
import logging

def pipeline(args):

    # step1 : preprocessing data, reformat, Fill missing data
    volume_feature, trajectory_feature, weather_feature = feature.PreprocessingRawdata(force_update=args.update_feature)
    data = feature.ReformatData(volume_feature, trajectory_feature, weather_feature)

    # step2 : training

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_feature', type=int, choices=[0, 1], default=0,
                        help='indicate whether to update feature')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # Start Pipeline
    logging.info(args)
    pipeline(args)



