import gzip
import pickle
import numpy as np
from scipy import stats
import matplotlib
# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import sys

from theano import tensor as T
from theano import config
from theano.compile.nanguardmode import NanGuardMode
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from data import loadSpiralData, loadMnistData
from GMVAE import GMVAE
from hyper import getHyper, loadHyper

def plot2D(name, data, model, clustered=True):
    """Expects data to be a dictionnary of samples, key = 0,1,2, ..., num_clust"""

    plt.figure(1)
    if clustered:
        for y_samples in data.values():
            plt.scatter(y_samples[:,0], y_samples[:,1])
    else:
        plt.scatter(data[:,0],data[:,1])

    plt.savefig(model.hyper['exp_folder']+'/'+name+'.png')
    plt.clf()

    if clustered:
            all_y_samples = np.concatenate(data.values(), axis=0)
    else:
        all_y_samples = data

    plt.figure(1)
    plt.hist2d(all_y_samples[:,0], all_y_samples[:,1], bins=200)
    plt.savefig(model.hyper['exp_folder']+'/'+name+'hist.png')
    plt.clf()

def plotMnist(name, data, model):
    i = 0
    f, _plots = plt.subplots(2, 9)
    for row in _plots:
        for column in row:
            column.imshow(data[i], cmap='gray')
            column.set_xlim([0,28])
            column.set_ylim([0,28])
            column.axis('off')
            i += 1
    plt.savefig(model.hyper['exp_folder']+'/'+name+'.png')
    plt.clf()

def plot_learning_curves(name, model):

    #[i,L_elbo_train, L_elbo_valid, z_prior_train, z_prior_valid]

    with gzip.open(model.hyper['exp_folder']+'/analysis.pkl', 'rb') as f:
        analysis = pickle.load(f)

    train, = plt.plot(analysis[:,0], analysis[:,1], label='L_elbo Train')
    valid, = plt.plot(analysis[:,0], analysis[:,2], label='L_elbo Valid')
    plt.legend(handles=[train, valid])

    plt.savefig(model.hyper['exp_folder']+'/curves1.png')
    plt.clf()

    train, = plt.plot(analysis[:,0], analysis[:,3], label='z-prior Train')
    valid, = plt.plot(analysis[:,0], analysis[:,4], label='z-prior Valid')
    plt.legend(handles=[train, valid])

    plt.savefig(model.hyper['exp_folder']+'/curves2.png')
    plt.clf()


def sample(model, n=500):
    """Sample from p (DAG presented in the GMVAE article) using ancestral sampling"""
    #TODO finish this function!

    print 'Sampling from p(y)...'

    w_samples = np.random.normal(0,1,size=(n,1,model.hyper['w_dim'])).astype(config.floatX)

    z_samples = np.random.randint(0,model.hyper['num_clust'],size=(n,))

    unique, counts = np.unique(z_samples, return_counts=True)
    count_dict = zip(unique, counts)
    count_dict += [(-1,0)]
    count_dict = dict(count_dict)
    sum_ = 0

    x_samples = dict([(k,[]) for k in range(model.hyper['num_clust'])])
    y_samples = dict([(k,[]) for k in range(model.hyper['num_clust'])])
    for cluster in unique:
        # shapes are (number of times z gave 'cluster', 1 , num of dimension of x)

        pxgzw_mus, pxgzw_vars = model.computePxgzwParams(cluster,w_samples[sum_: sum_ + count_dict[cluster]])

        sum_ += count_dict[cluster]

        for i in range(pxgzw_mus.shape[0]):
            x_samples[cluster] += [np.random.multivariate_normal(pxgzw_mus[i,0,:], pxgzw_vars[i,0,:]*np.eye(model.hyper['x_dim'])).astype(config.floatX)]

        # shape is (number of times z gave 'cluster',1, x_dim)
        x_samples_cluster = np.array(x_samples[cluster])
        print x_samples_cluster.shape
        x_samples_cluster = x_samples_cluster.reshape(x_samples_cluster.shape[0],1,x_samples_cluster.shape[1])

        if model.hyper['mode'] == 'spiral':
            # shapes are (number of times z gave 'cluster',1, y_dim)
            pygx_mu, pygx_var = model.computePygxParams(x_samples_cluster)

            for i in range(pxgzw_mus.shape[0]):
                y_samples[cluster] += [np.random.multivariate_normal(pygx_mu[i,0,:], pygx_var[i,0,:]*np.eye(model.hyper['y_dim'])).astype(config.floatX)]

            y_samples[cluster] = np.array(y_samples[cluster])
        elif model.hyper['mode'] == 'mnist':
            pygx_mu = model.computePygxParams(x_samples_cluster)
            y_samples[cluster] += [pygx_mu]

    with open(model.hyper['exp_folder']+'/samples.pkl','wb') as f:
        pickle.dump(y_samples,f)

    return y_samples


def train(model):
    """"""
    print 'Training ...'

    L_elbo_train, L_elbo_modif_train, z_prior_train, z_prior_modif_train = model.computeMetricsTrain()
    L_elbo_valid, L_elbo_modif_valid, z_prior_valid, z_prior_modif_valid = model.computeMetricsValid()

    print '\nBefore learning:'
    print 'Training L_elbo : {}'.format(L_elbo_train)
    print 'Validation L_elbo: {}'.format(L_elbo_valid)
    print 'Training L_elbo modified : {}'.format(L_elbo_modif_train)
    print 'Validation L_elbo modified: {}'.format(L_elbo_modif_valid)
    print 'Training z-prior term : {}'.format(z_prior_train)
    print 'Validation z-prior term: {}'.format(z_prior_valid)
    print 'Training z-prior modified term: {}'.format(z_prior_modif_train)
    print 'Validation z-prior modified term: {}'.format(z_prior_modif_valid)

    #Analysis table (for plots!)
    analysis = np.array([[0]*5],dtype=config.floatX) # columns: epoch number, self.L_elbo (train), idem (valid), self.z_prior (train), idem (valid)

    # Number of batch per epoch
    nb_batch = model.train_data.get_value(borrow=True).shape[0] // model.hyper['batch_size']

    # Init best_valid_error
    best_L_elbo_valid = -np.inf

    # Init patience_count
    patience_count = 0

    i = 0
    while i < model.hyper['max_epoch'] and patience_count < model.hyper['patience']:

        for batch_idx in np.arange(0,nb_batch):

            model.trainModel(batch_idx)

        i += 1

        if i % model.hyper['valid_freq'] == 0:
            #self.L_elbo, self.L_elbo_modif, T.mean(self.z_prior), T.mean(self.z_prior_modif)

            L_elbo_train, L_elbo_modif_train, z_prior_train, z_prior_modif_train = model.computeMetricsTrain()
            L_elbo_valid, L_elbo_modif_valid, z_prior_valid, z_prior_modif_valid = model.computeMetricsValid()

            analysis = np.concatenate([analysis, [[i,L_elbo_train, L_elbo_valid, z_prior_train, z_prior_valid]]], axis=0)

            if best_L_elbo_valid < L_elbo_valid:
                best_L_elbo_valid = L_elbo_valid
                model.saveParams()
                patience_count = 0
            else:
                patience_count += 1

            print '\nAfter {} completed epochs:'.format(i)
            print 'Training L_elbo : {}'.format(L_elbo_train)
            print 'Validation L_elbo: {}'.format(L_elbo_valid)
            print 'Training L_elbo modified : {}'.format(L_elbo_modif_train)
            print 'Validation L_elbo modified: {}'.format(L_elbo_modif_valid)
            print 'Training z-prior term : {}'.format(z_prior_train)
            print 'Validation z-prior term: {}'.format(z_prior_valid)
            print 'Training z-prior modified term: {}'.format(z_prior_modif_train)
            print 'Validation z-prior modified term: {}'.format(z_prior_modif_valid)
            print '-' * 80

    #Removing the first line of the table (only zeros in it)
    analysis = analysis[1:]

    # Saving a compressed version of the analysis tables
    with gzip.open(model.hyper['exp_folder']+'/analysis.pkl','wb') as f:
        pickle.dump(analysis, f)


if __name__ == '__main__':
    if sys.argv[1] == '-spiral':
        hyper = getHyper('spiral')
    elif sys.argv[1] == '-mnist':
        hyper = getHyper('mnist')
    else:
        print('Usage: python main.py {-spiral or -mnist}')
        sys.exit()

    print "Hyper Params: "
    print hyper

    gm_vae = GMVAE(hyper)
    gm_vae.buildGraph()

    if hyper['mode'] == 'spiral':
        train_data, valid_data = loadSpiralData(hyper)
        plot2D('train_data', train_data.get_value(), gm_vae, clustered=False)
    elif hyper['mode'] == 'mnist':
        train_data, valid_data = loadMnistData(hyper)
        plotMnist('train_data', train_data.get_value(), gm_vae)

    gm_vae.compile(train_data, valid_data)

    samples = sample(gm_vae)
    #with open(hyper['exp_folder']+'/samples.pkl','rb') as f:
    #   samples = pickle.load(f)

    if hyper['mode'] == 'spiral':
        plot2D('samples_before',samples, gm_vae)
    elif hyper['mode'] == 'mnist':
        plotMnist('samples_before',samples, gm_vae)

    train(gm_vae)

    gm_vae.setBestParams()
    samples = sample(gm_vae)
    if hyper['mode'] == 'spiral':
        plot2D('samples_after_training',samples, gm_vae)
    elif hyper['mode'] == 'mnist': 
        plotMnist('samples_after_training',samples, gm_vae)

    plot_learning_curves('learn_curves', gm_vae)
