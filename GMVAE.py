import gzip
import pickle

import numpy as np
from scipy import stats
import matplotlib
# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from theano import config
from theano.compile.nanguardmode import NanGuardMode
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from blocks.bricks import MLP, Rectifier, Logistic, BatchNormalization
from blocks.bricks.conv import Convolutional
from blocks.roles import PARAMETER, WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import Constant, Uniform

from optimizers import adam, sgd, adagrad

rng = np.random.RandomState(12345)

class GMVAE(object):

    def __init__(self,hyper):
        self.hyper = hyper
        self.srng = RandomStreams(seed=234)

    def buildQ(self):
        """Build the graph for the Q network"""

        print 'Building Q ...'

        if self.hyper['mode'] == 'spiral':
            self.y = T.matrix('y')

            mlp = MLP(activations=self.hyper['q_activs'],
                      dims=self.hyper['q_dims'],
                      weights_init=self.hyper['q_W_init'],
                      biases_init=Constant(0))

            q_parameters = mlp.apply(self.y)
            mlp.initialize()
        elif self.hyper['mode'] == 'mnist':
            self.y = T.matrix('y')

            bn1 = BatchNormalization(input_dim=(1, 28, 28), broadcastable=(False, True, True))
            cn1 = Convolutional(filter_size=(5, 5),
                               num_filters=16,
                               num_channels=1,
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_1')
            rect1 = Rectifier().apply(cn1.apply(bn1.apply(self.y.reshape([self.y.shape[0], 1, 28, 28]))))
            #rect1 = cn1.apply(self.y.reshape([self.y.shape[0], 1, 28, 28]))

            bn2 = BatchNormalization(input_dim=(16, 24, 24), broadcastable=(False, True, True))
            cn2 = Convolutional(filter_size=(5, 5),
                               num_filters=32,
                               num_channels=16,
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_2')
            rect2 = Rectifier().apply(cn2.apply(bn2.apply(rect1)))
            #rect2 = cn2.apply(rect1)

            bn3 = BatchNormalization(input_dim=(32, 20, 20), broadcastable=(False, True, True))
            cn3 = Convolutional(filter_size=(4, 4),
                               num_filters=64,
                               num_channels=32,
                               step=(2, 2),
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_3')
            rect3 = Rectifier().apply(cn3.apply(bn3.apply(rect2)))
            #rect3 = cn3.apply(rect2)

            bn4 = BatchNormalization(input_dim=(64, 9, 9), broadcastable=(False, True, True))
            mlp = MLP(activations=[Rectifier(), None],
                      dims=[5184, 500, 2*self.hyper['x_dim'] + 2*self.hyper['w_dim']],
                      weights_init=self.hyper['q_W_init'],
                      biases_init=Constant(0))
            q_parameters = mlp.apply(bn4.apply(rect3).reshape([rect3.shape[0], 5184]))

            bn1.initialize()
            cn1.initialize()
            bn2.initialize()
            cn2.initialize()
            bn3.initialize()
            cn3.initialize()
            bn4.initialize()
            mlp.initialize()

        # self.qxgy_mu.shape == (minibatch size, num of dimension of x)
        self.qxgy_mu = q_parameters[:,:self.hyper['x_dim']]

        # self.qxgy_var.shape == (minibatch size, num of dimension of x)
        self.qxgy_var = T.exp( q_parameters[:,self.hyper['x_dim']:2*self.hyper['x_dim']] )

        # self.qwgy_mu.shape == (minibatch size, num of dimension of w)
        self.qwgy_mu = q_parameters[:,2*self.hyper['x_dim']:2*self.hyper['x_dim']+self.hyper['w_dim']]

        # self.qwgy_var.shape == (minibatch size, num of dimension of w)
        self.qwgy_var = T.exp( q_parameters[:,2*self.hyper['x_dim']+self.hyper['w_dim']:] )


        #---Will be useful to compute samples from q(x|y)---#
        #self.eps_x.shape == (minibatch size, # of x samples , # of dimension of x)
        self.eps_x = self.srng.normal((self.qxgy_mu.shape[0] ,self.hyper['L_x'] ,self.hyper['x_dim']))

        #self.x corresponds roughly to the function g(\epsilon,y) (see reparametrization trick in Kingma 2014)
        #self.x.shape == (minibatch size, # of x samples , # of dimension of x)
        self.x = self.qxgy_mu.dimshuffle(0,'x',1) + T.sqrt(self.qxgy_var).dimshuffle(0,'x',1)*self.eps_x

        #---Will be useful to compute samples from q(w|y)---#
        #self.eps_w.shape == (minibatch size, # of w samples , # of dimension of w)
        self.eps_w = self.srng.normal((self.qwgy_mu.shape[0] ,self.hyper['L_w'] ,self.hyper['w_dim']))

        #self.w corresponds roughly to the function g(\epsilon,y) (see reparametrization trick in Kingma 2014)
        #self.w.shape == (minibatch size, # of w samples , # of dimension of w)
        self.w = self.qwgy_mu.dimshuffle(0,'x',1) + T.sqrt(self.qwgy_var).dimshuffle(0,'x',1)*self.eps_w


        #---Building the log density q(x|y)---#
        little_num = 10**(-32)
        inside_exp = -T.sum((self.x - self.qxgy_mu.dimshuffle(0,'x',1))**2/(2*self.qxgy_var.dimshuffle(0,'x',1)), axis=2)
        norm_cst =  (2*np.pi)**(-self.hyper['x_dim']/2.)*T.exp(T.sum(T.log(self.qxgy_var), axis=1))**(-1/2.)

        # shape == (minibatch size, # of x samples)
        qxgy = norm_cst.dimshuffle(0,'x')*T.exp(inside_exp)

        # shape == (minibatch size, # of x samples)
        self.log_qxgy = T.log(qxgy + little_num)


    def buildP(self):
        """Build the graph related to P's networks"""

        print 'Building P ...'

        #---Building p(y|x)---#
        if self.hyper['mode'] == 'spiral':
            pygx_params_mlp = MLP(activations=self.hyper['pygx_activs'],
                              dims=self.hyper['pygx_dims'],
                              weights_init=self.hyper['pygx_W_init'],
                              biases_init=Constant(0))

            pygx_params = pygx_params_mlp.apply(self.x.reshape((self.x.shape[0]*self.x.shape[1],self.x.shape[2])))
            pygx_params = pygx_params.reshape((self.x.shape[0], self.x.shape[1], 2*self.hyper['y_dim']))
            pygx_params_mlp.initialize()
        elif self.hyper['mode'] == 'mnist':
            bn5 = BatchNormalization(input_dim=200)
            pygx_params_mlp = MLP(activations=[Rectifier(), None],
                                  dims=[200, 500, 5476],
                                  weights_init=self.hyper['pygx_W_init'],
                                  biases_init=Constant(0))
            rect5 = Rectifier().apply(pygx_params_mlp.apply(bn5.apply(self.x.reshape((self.x.shape[0]*self.x.shape[1],self.x.shape[2])))))
            #rect5 = pygx_params_mlp.apply(self.x.reshape((self.x.shape[0]*self.x.shape[1],self.x.shape[2])))

            bn6 = BatchNormalization(input_dim=(1, 74, 74), broadcastable=(False, True, True))
            cn6 = Convolutional(filter_size=(4, 4),
                               num_filters=64,
                               num_channels=1,
                               step=(2, 2),
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_7')
            rect6 = Rectifier().apply(cn6.apply(bn6.apply(rect5.reshape([rect5.shape[0], 1, 74, 74]))))
            #rect7 = cn7.apply(rect5.reshape([rect5.shape[0], 1, 41, 41]))

            bn7 = BatchNormalization(input_dim=(64, 36, 36), broadcastable=(False, True, True))
            cn7 = Convolutional(filter_size=(5, 5),
                               num_filters=32,
                               num_channels=64,
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_8')
            rect7 = Rectifier().apply(cn7.apply(bn7.apply(rect6)))
            #rect8 = cn8.apply(rect7)

            bn8 = BatchNormalization(input_dim=(32, 32, 32), broadcastable=(False, True, True))
            cn8 = Convolutional(filter_size=(5, 5),
                               num_filters=16,
                               num_channels=32,
                               weights_init=Uniform(mean=0, width=.5),
                               biases_init=Constant(0),
                               name='conv_9')
            #pygx_params = Rectifier().apply(cn9.apply(rect8))
            rect8 = Rectifier().apply(cn8.apply(bn8.apply(rect7)))

            last_mlp = MLP(activations=[None],
                          dims=[12544, 784],
                          weights_init=self.hyper['pygx_W_init'],
                          biases_init=Constant(0))

            pygx_params = last_mlp.apply(rect8.reshape((self.x.shape[0], 12544)))
            pygx_params = pygx_params.reshape((self.x.shape[0], self.x.shape[1], self.hyper['y_dim']))

            bn5.initialize()
            pygx_params_mlp.initialize()
            bn6.initialize()
            cn6.initialize()
            bn7.initialize()
            cn7.initialize()
            bn8.initialize()
            cn8.initialize()
            last_mlp.initialize()

        if self.hyper['mode'] == 'spiral':
            # self.pygx_mu.shape == (minibatch size, L_x , num of dimension of y)
            self.pygx_mu = pygx_params[:,:,:self.hyper['y_dim']]
            # self.pygx_var.shape == (minibatch size, L_x, num of dimension of y)
            self.pygx_var = T.exp(pygx_params[:,:,self.hyper['y_dim']:])
        elif self.hyper['mode'] == 'mnist':
            self.pygx_mu = T.nnet.nnet.sigmoid(pygx_params[:,:,:self.hyper['y_dim']])
            #self.pygx_var = 0.1 * T.ones(pygx_params[:,:,self.hyper['y_dim']:].shape, dtype='float')
            #self.pygx_var = T.exp(pygx_params[:,:,self.hyper['y_dim']:])

        #---Building graph for the density of p(y|x)---#
        #little_num = 10**(-7)
        #inside_exp = -T.sum((self.y.dimshuffle(0,'x',1) - self.pygx_mu)**2/(2*self.pygx_var), axis=2)
        #norm_cst =  (2*np.pi)**(-self.hyper['y_dim']/2.)*T.exp(T.sum(T.log(self.pygx_var), axis=2))**(-1/2.)
        if self.hyper['mode'] == 'spiral':
            little_num = 10**(-32)
            inside_exp = -T.sum((self.y.dimshuffle(0,'x',1) - self.pygx_mu)**2/(2*self.pygx_var), axis=2)
            norm_cst =  (2*np.pi)**(-self.hyper['y_dim']/2.)*T.exp(T.sum(T.log(self.pygx_var), axis=2))**(-1/2.)
                                                            
            # shape == (minibatch size, # of x samples)
            pygx = norm_cst*T.exp(inside_exp)

            # shape == (minibatch size, # of x samples)
            self.log_pygx = T.log(pygx + little_num)
        elif self.hyper['mode'] == 'mnist':
            little_num = 10**(-7)
            self.pygx_mu = T.clip(self.pygx_mu, little_num, 1.0 - little_num)
            self.log_pygx = T.sum(self.y.dimshuffle(0, 'x', 1) * T.log(self.pygx_mu) + (1 - self.y.dimshuffle(0, 'x', 1)) * T.log(1 - self.pygx_mu), axis=2)

        # shape == (minibatch size, # of x samples)
        #pygx = norm_cst*T.exp(inside_exp)

        # shape == (minibatch size, # of x samples)
        #self.log_pygx = T.log(pygx + little_num)

        #---Building NN for p(x|z=j,w) for all j---#
        pxgzw_mus = [None]*self.hyper['num_clust']
        pxgzw_vars = [None]*self.hyper['num_clust']
        pxgzw = [None]*self.hyper['num_clust']

        for j in range(self.hyper['num_clust']):

            pxgzw_params_mlp = MLP(activations=self.hyper['pxgzw_activs'][j],
                      dims=self.hyper['pxgzw_dims'][j],
                      weights_init=self.hyper['pxgzw_W_init'],
                      biases_init=Constant(0))

            pxgzw_params = pxgzw_params_mlp.apply(self.w.reshape((self.w.shape[0]*self.w.shape[1],self.w.shape[2])))
            pxgzw_params = pxgzw_params.reshape((self.w.shape[0],self.w.shape[1], 2*self.hyper['x_dim']))
            pxgzw_params_mlp.initialize()

            # pxgzw_mus[j].shape == (minibatch size, L_w , num of dimension of x)
            pxgzw_mus[j] = pxgzw_params[:,:,:self.hyper['x_dim']]

            # pxgzw_vars[j].shape == (minibatch size, L_w, num of dimension of x)
            pxgzw_vars[j] = T.exp( pxgzw_params[:,:,self.hyper['x_dim']:] )

            #---Building graph for the density of p(x|z=j,w)---#
            little_num = 10**(-32)
            inside_exp = -T.sum((self.x.dimshuffle(0,'x',1,2) - pxgzw_mus[j].dimshuffle(0,1,'x',2))**2/(2*pxgzw_vars[j].dimshuffle(0,1,'x',2)), axis=3)
            norm_cst =  (2*np.pi)**(-self.hyper['x_dim']/2.)*T.exp(T.sum(T.log(pxgzw_vars[j]), axis=2))**(-1/2.)

            # shape == (minibatch size, # of w samples (L_w), # of x samples (L_x))
            pxgzw[j] = norm_cst.dimshuffle(0,1,'x')*T.exp(inside_exp)


        # shape is (minibatch size, L_w , # of clusters , num of dimension of x)
        self.pxgzw_mus = T.concatenate([mu.dimshuffle(0,1,'x',2) for mu in pxgzw_mus], axis=2)
        # shape is (minibatch size, L_w , # of clusters , num of dimension of x)
        self.pxgzw_vars = T.concatenate([var.dimshuffle(0,1,'x',2) for var in pxgzw_vars], axis=2)

        # self.pxgzw.shape == (minibatch size, L_w, L_x, num_clust)
        self.pxgzw = T.concatenate([density.dimshuffle(0,1,2,'x') for density in pxgzw], axis=3)
        self.log_pxgzw = T.log(self.pxgzw + little_num)

        #---Building the p(z=j|x,w) posterior for all j---#
        # self.log_pzgxw.shape == (minibatch size, L_w, L_x, num_clust)
        self.log_pzgxw = T.log(self.pxgzw + little_num) -T.log(T.sum(self.pxgzw + little_num, axis=3).dimshuffle(0,1,2,'x'))


    def buildReconstructionTerm(self):
        """Reconstruction term (see GMVAE article's terminology)"""

        # shape is (1,)
        self.reconst = T.mean(self.log_pygx)

    def buildConditionalPriorTerm(self):
        """Conditional prior term (see GMVAE article's terminology)
        There's an error in the article regarding the order of integration..."""

        # shape is (1,)
        self.conditional_prior = - T.mean(T.sum(T.exp(self.log_pzgxw)*(self.log_qxgy.dimshuffle(0,'x',1,'x') - self.log_pxgzw), axis=3))

    def buildWPriorTerm(self):
        """w-prior term (see GMVAE article's terminology)"""

        # self.w_prior.shape == (1,)
        self.w_prior = T.mean(0.5*T.sum(1 + T.log(self.qwgy_var) - self.qwgy_mu**2-self.qwgy_var, axis=1))

        self.w_prior_modif = - T.maximum(self.hyper['treshold_w_prior'], -self.w_prior)

    def buildZPriorTerm(self):
        """z-prior term (see GMVAE article's terminology)"""

        # shape is (1,)
        self.z_prior = - T.mean(T.sum(T.exp(self.log_pzgxw)*(self.log_pzgxw + T.log(self.hyper['num_clust'])), axis=3))

        self.z_prior_modif = - T.maximum(self.hyper['treshold_z_prior'], - self.z_prior)


    def buildObjective(self):
        """Builds the approximate objective corresponding to L_elbo in GMVAE article"""

        # self.z_prior might be the modified version
        self.L_elbo = self.reconst + self.conditional_prior + self.w_prior + self.z_prior

        self.L_elbo_modif = self.reconst + self.conditional_prior + self.w_prior_modif + self.z_prior_modif

        #---Getting model parameter---#
        cg = ComputationGraph(self.L_elbo)
        #self.phi_theta is the list of all the parameters in q and p.
        self.params = VariableFilter(roles=[PARAMETER])(cg.variables)


    def buildGrad(self):
        """Builds the approx gradient"""

        self.grads = T.grad(self.L_elbo_modif, self.params)


    def buildGraph(self):
        """Builds the whole graph"""

        print 'Building graph...'

        self.buildQ()
        self.buildP()
        self.buildReconstructionTerm()
        self.buildConditionalPriorTerm()
        self.buildWPriorTerm()
        self.buildZPriorTerm()

        self.buildObjective()
        self.buildGrad()

    def compile(self, train_data, valid_data):
        """Compiles all the necessary theano functions"""

        print 'Compiling theano functions...'

        # Training functions
        #train_input = T.matrix('train_input')
        #train_loss_x = T.matrix('train_loss_x')
        #valid_loss_x = T.matrix('valid_loss_x')
        bs = self.hyper['batch_size']

        updates = eval(self.hyper['algo'])
        #TODO fix this 
        self.train_data = train_data
        self.valid_data = valid_data
        self.trainModel = theano.function(inputs=[self.y],
                                      outputs=self.grads,
                                      updates=updates)#,
                                      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

        # Metric computers (Have to be called in this order: self.computeMetricsTrain then self.computeMetricsValid)
        self.computeMetricsTrain = theano.function(inputs=[self.y],
                                      outputs=[self.L_elbo, self.L_elbo_modif, self.z_prior, self.z_prior_modif],
                                      no_default_updates=True)#, #To be sure that the samples are the same when calling self.computeMetricsValid and self.computeMetricsTrain
                                      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

        self.computeMetricsValid = theano.function(inputs=[self.y],
                                      outputs=[self.L_elbo, self.L_elbo_modif, self.z_prior, self.z_prior_modif])#,
                                      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

        z_index = T.iscalar('z_index')
        self.computePxgzwParams = theano.function(inputs=[z_index, self.w],
                                      outputs=[self.pxgzw_mus[:,:,z_index,:],self.pxgzw_vars[:,:,z_index,:]])#,
                                      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

        if self.hyper['mode'] == 'spiral':
            self.computePygxParams = theano.function(inputs=[self.x], 
                                        outputs=[self.pygx_mu,self.pygx_var])#,
                                        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        elif self.hyper['mode'] == 'mnist':
            self.computePygxParams = theano.function(inputs=[self.x], outputs=self.pygx_mu)

    def saveParams(self):

        with open(self.hyper['exp_folder']+'/bestParams.pkl', 'wb') as f:
            pickle.dump(self.params,f)

    def setBestParams(self):

        print "Setting best parameters"

        with open(self.hyper['exp_folder']+'/bestParams.pkl', 'rb') as f:
            best_params = pickle.load(f)

        for best_param, param in zip(best_params, self.params):
            param.set_value(best_param.get_value())
