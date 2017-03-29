from config import cfg
import util
import feature

if __name__ == '__main__':
    # step1 : preprocessing data, reformat, Fill missing data
    volume_feature, trajectory_feature = feature.PreprocessingRawdata(force_update=True)
    data = feature.ReformatData(volume_feature, trajectory_feature)

    # step2 : training
