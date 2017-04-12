## KDD Cup 2017


#### Project Organization

- ```bnlstm```: An implementation of Recurrent Batch Normalization in TensorFlow
- ```config```: Model config and dataset description
- ```data.checkpoint```: Tensorflow model files
- ```data.dataSets```:  Rawdata, in csv format
- ```data.features```:  Temporal data, in json format
- ```data.prediction```: Prediction results
- ```dataloader```: Data iterator
- ```feature```: Functions concern feature preprocessing
- ```model```: Machine learning models
- ```util```: I/O and other utility functions

#### Requirements
- Tensorflow 1.0
- python 3.5

#### Get started

- Download [dataset](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100068.5678.2.Uwgmr3&raceId=231597), unpack and move them to ```data.dataSets``` directory
- Check and edit all the fields related to dataset and experiment settings in ```config.py```


#### References
1. Cooijmans T, Ballas N, Laurent C, et al. Recurrent batch normalization[J]. arXiv preprint arXiv:1603.09025, 2016.
2. Shahsavari B, Abbeel P. Short-term traffic forecasting: Modeling and learning spatio-temporal relations in transportation networks using graph neural networks[J]. 2015.
3. Della Valle E, Celino I, Dell’Aglio D, et al. Urban Computing: a challenging problem for Semantic Technologies[C]//2nd International Workshop on New Forms of Reasoning for the Semantic Web (NEFORS 2008) co-located with the 3rd Asian Semantic Web Conference (ASWC 2008). 2008.
4. Che Z, Purushotham S, Cho K, et al. Recurrent neural networks for multivariate time series with missing values[J]. arXiv preprint arXiv:1606.01865, 2016.

