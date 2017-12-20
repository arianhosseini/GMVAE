
import numpy as np
import struct
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
def loadMnistData(hyper):
    print 'Loading data...'
    with open('data/train-images-idx3-ubyte', 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data_train = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    with open('data/train-images-idx3-ubyte', 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data_valid = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    data_train = (data_train/255.).astype(np.float32)
    data_valid = (data_valid/255.).astype(np.float32)

    data_train = data_train[:6000]
    data_valid = data_valid[:1000]

    data_train = data_train.reshape((data_train.shape[0], data_train.shape[1] * data_train.shape[2]))
    data_valid = data_valid.reshape((data_valid.shape[0], data_valid.shape[1] * data_valid.shape[2]))
    #data_train = theano.shared(data_train.astype(config.floatX), borrow=True)
    #data_valid = theano.shared(data_valid.astype(config.floatX), borrow=True)

    data_train = data_train[:6000]
    data_valid = data_valid[:1000]

    return data_train, data_valid
