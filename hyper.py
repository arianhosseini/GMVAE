import os

import pickle
import numpy as np

from blocks.bricks import Rectifier, Logistic, Softmax, Tanh
from blocks.initialization import Uniform, IsotropicGaussian

def getHyperBasic():

    hyper = {}





    return hyper
def update_hyper_by_args(hyper, args):
    for name in sorted(vars(args)):
        hyper[name] = getattr(args, name)
    return hyper

def getHyperSpiral(args):

    hyper = {}

    hyper['w_dim'] = 2
    hyper['x_dim'] = 2
    hyper['y_dim'] = 2
    hyper['num_clust'] = 5 # Number of clusters

    #---P related networks architecture---#
    #---p(y|x)---#
    hyper['pygx_activs'] = [Rectifier(), Rectifier(), None]
    hyper['pygx_dims'] = [hyper['x_dim'], 120, 120, 2*hyper['y_dim']]
    #width_unif = 2*np.sqrt(6./(hyper['enc_dims'][0] + hyper['enc_dims'][-1]))
    hyper['pygx_W_init'] = Uniform(mean=0, width=.8)
#IsotropicGaussian(std=2.*np.sqrt(0.01), mean=0)

    #---p(x|z=j,w) for all j---#
    hyper['pxgzw_activs'] = [None]*hyper['num_clust']
    hyper['pxgzw_dims'] = [None]*hyper['num_clust']
    for j in range(hyper['num_clust']):
        hyper['pxgzw_activs'][j] = [Tanh(), None]
        hyper['pxgzw_dims'][j] = [hyper['w_dim'], 120, 2*hyper['x_dim']]
    hyper['pxgzw_W_init'] = Uniform(mean=0, width=.8)
#IsotropicGaussian(std=np.sqrt(0.01), mean=0)

    #---Q related networks architecture---#
    #---q(x|y) and q(w|y)---#
    # Both distributions' parameters are outputted by the same NN (see appendix of GMVAE article)
    hyper['q_activs'] = [Rectifier(), Rectifier(), None]
    hyper['q_dims'] = [hyper['y_dim'], 120, 120, 2*hyper['x_dim']+2*hyper['w_dim']]
    #width_unif= 2*np.sqrt(6./(hyper['dec_dims'][0] + hyper['dec_dims'][-1]))
    hyper['q_W_init'] = IsotropicGaussian(std=np.sqrt(0.01), mean=0)

    #---Optimization related---#
    hyper['algo'] = 'adam(self.params, self.grads, self.hyper[\'lr\'])'

    hyper['lr'] = 0.001
    hyper['batch_size'] = 100
    hyper['max_epoch'] = 100000
    hyper['patience'] = 50 # Patience of hyper['patience'] (measured in validation checks...)
    hyper['valid_freq'] = 10 # Will check the valid "error" every ___ epochs

    hyper['L_w'] =  1 # Number of w samples for each examples of a minibatch.
    hyper['L_x'] = 1 # Number of x samples for each examples of a minibatch.

    hyper['exp_folder'] = 'exp_zth_1.4'

    hyper['normalize_data'] = True
    hyper['threshold_z_prior'] = 1.4 # TODO: What's the value they're using in GMVAE??
    hyper['threshold_w_prior'] = 0

    hyper = update_hyper_by_args(hyper, args) #overwrite some params by args

    # Saving hyper in exp_folder
    if not os.path.exists(hyper['exp_folder']):
        os.makedirs(hyper['exp_folder'])

    with open(hyper['exp_folder']+'/hyper_params.txt','w') as f:
        for key,value in sorted(hyper.iteritems()):
            if type(value) in [int,float,str,bool]:
                f.write(str(key) + ": " + str(value)+"\n")

    with open(hyper['exp_folder']+'/hyper.pkl','wb') as f:
        pickle.dump(hyper,f)

    return hyper

def loadHyper(exp_folder):

	with open(exp_folder+'/hyper.pkl','rb') as f:

		hyper = pickle.load(f)

	return hyper
