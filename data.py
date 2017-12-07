
import numpy as np
from scipy import stats
import theano
from theano import tensor as T
from theano import config

from fakedata import make_pinwheel_data

def loadSpiralData(hyper):
    """
    choice can be:
        - 'spiral'
        - 'mnist'
        - ... etc
    """
    print 'Loading data...'
    num_clusters = 5           # number of clusters in pinwheel data
    samples_per_cluster = 2000  # number of samples per cluster in pinwheel
    # generate synthetic data
    data_train = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)
    data_valid = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

    if hyper['normalize_data']:
        data_train = stats.mstats.zscore(data_train, axis = 0)
        data_valid = stats.mstats.zscore(data_valid, axis = 0)

    n_train = data_train.shape[0]
    n_valid = data_valid.shape[0]
    data_train = theano.shared(data_train.astype(config.floatX), borrow=True)
    data_valid = theano.shared(data_valid.astype(config.floatX), borrow=True)
    return data_train, data_valid
def loadMnist(hyper):
    raise NotImplementedError("")
