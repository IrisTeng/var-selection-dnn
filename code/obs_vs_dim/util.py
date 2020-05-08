import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import GPy
import seaborn as sns
import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


## toy data

def sin_toy(n_obs, dim_in, sig2=.01, seed=0):
    r = np.random.RandomState(seed)  
    Z = r.uniform(-5,5,(n_obs, dim_in))
    f = lambda x: np.sin(x)
    Y = (f(Z[:,0]) + r.normal(0,sig2,n_obs)).reshape(-1,1)
    
    return Z, Y.reshape(-1,1), sig2

def rbf_toy(n_obs, dim_in, sig2=.01, seed_f=8, seed_zy=0):

    r_f = np.random.RandomState(seed_f)  
    r_zy = np.random.RandomState(seed_zy)  
    
    # sample 1D function
    n_obs_sample = 500
    sig2_f = .1
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    Z = r_f.uniform(-5,5,(n_obs_sample,1))
    f = r_f.multivariate_normal(np.zeros(n_obs_sample), kernel.K(Z), 1).reshape(-1,1)
    Y = f + r_f.normal(0,sig2_f,(n_obs_sample,1))
    
    # fit gp to this function
    m = GPy.models.GPRegression(Z,Y,kernel)
    m.Gaussian_noise.variance = sig2_f
    
    # sample Z data that is actually used (not using same random state)
    # and get Y data by predicting using fitted GP
    Z = r_zy.uniform(-5, 5, (n_obs, dim_in))
    Y,_ = m.predict(Z[:,0].reshape(-1,1)) + r_zy.normal(0,sig2,(n_obs,1))

    return Z, Y.reshape(-1,1), sig2

def bkmr_toy(n_obs, dim_in, sig2=.5):
    '''
    generates toy data by running SimData function from bkrm R package
    requires rpy2 to be installed
    Note: random seed not implemented
    '''
    r = robjects.r
    bkmr = importr('bkmr') 

    n = robjects.IntVector([n_obs])
    M = robjects.IntVector([dim_in])
    sigsq_true = robjects.FloatVector([sig2])

    out = bkmr.SimData(n=n, M=M, beta_true=0, sigsq_true = sigsq_true, Zgen="realistic")

    Z = np.asarray(out.rx2['Z'])
    Y = np.asarray(out.rx2['y'])
    
    return Z, Y.reshape(-1,1), sig2

def workshop_data(n_obs_samp=None, dim_in_samp=None, dir_in='../../data/workshop', seed=0):
    '''
    randomly samples observations and mixture components. 
    by default uses all observations and mixture components.

    used in bkmr as:
    kmbayes(y=lnLTL_z, Z=lnmixture_z, X=covariates)
    ''' 

    Z=pd.read_csv(os.path.join(dir_in, 'Z.csv'), header=0).to_numpy() # lnmixture_z
    X=pd.read_csv(os.path.join(dir_in, 'X.csv'), header=0).to_numpy() # covariates
    Y=pd.read_csv(os.path.join(dir_in, 'y.csv'), header=0).to_numpy() # lnLTL_z

    # randomly sample observations
    if n_obs_samp is not None:
        r = np.random.RandomState(seed) 
        obs_keep = r.choice(Z.shape[0], n_obs_samp, replace=False)
        Z = Z[obs_keep,:]
        X = X[obs_keep,:]
        Y = Y[obs_keep]

    # randomly sample mixture components
    if dim_in_samp is not None:
        r = np.random.RandomState(seed+1) 
        Z = Z[:,r.choice(Z.shape[1], dim_in_samp, replace=False)]

    return Z, X, Y.reshape(-1,1)
    

def load_data(toy_name, n_obs, dim_in, seed=0):
    '''
    returns:
        Z: mixture components
        X: covariates
        Y: outcome
        sig2: observation variance
    '''
    if toy_name == 'sin':
        Z, Y, sig2 = sin_toy(n_obs, dim_in, seed=seed)
        X = None
    elif toy_name == 'rbf':
        Z, Y, sig2 = rbf_toy(n_obs, dim_in, seed_zy=seed)
        X = None
    elif toy_name == 'bkmr':
        Z, Y, sig2 = rbf_toy(n_obs, dim_in)
        X = None
    elif toy_name == 'workshop':
        Z, X, Y = workshop_data(n_obs, dim_in, seed=seed)
        sig2 = None
    else:
        print('toy not recognized')

    return Z, X, Y, sig2

## posterior metrics

def rmse(f, f_pred):
    # predictive root mean squared error (RMSE)
    return np.sqrt(np.mean((f - f_pred)**2))

def picp(f, f_pred_lb, f_pred_ub):
    # prediction interval coverage (PICP)
    return np.mean(np.logical_and(f >= f_pred_lb, f <= f_pred_ub))

def mpiw(f_pred_lb, f_pred_ub):
    # mean prediction interval width (MPIW)
    return np.mean(f_pred_ub - f_pred_lb)

def test_log_likelihood(mean, cov, test_y):
    return multivariate_normal.logpdf(test_y.reshape(-1), mean.reshape(-1), cov)


## plotting

def plot_results_grid(data, dim_in_list, n_obs_list, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,data.shape[2], figsize=(16,4), sharex=True, sharey=True)
    
    vmax = np.nanmax(data)
    ax[0].set_ylabel('num. obs.')
    for i in range(data.shape[2]):
        pcm=ax[i].imshow(data[:,:,i],vmin=0, vmax=vmax)
        ax[i].set_title('X_%d'%i)
        ax[i].set_xlabel('num. inputs')

    plt.xticks(np.arange(len(dim_in_list)), labels=dim_in_list)

    plt.yticks(np.arange(len(n_obs_list)), labels=n_obs_list)

    fig.colorbar(pcm, ax=ax[:], shrink=0.6)

    return fig, ax 


def plot_results_dist(data, dim_in, n_obs_list, data_true=None, fig=None, ax=None):
    '''
    data:   (obs x rep x input) <-- data for models with same input dimension
    data_true:   (input)

    For each combination of n_obs and variable_idx, plots distribution of psi_mean over reps.

    '''

    if fig is None and ax is None:
        fig,ax = plt.subplots(len(n_obs_list),1,figsize=(dim_in*2,16),sharex=True,sharey=True)

    ax[-1].set_xlabel('variable')
    ax[0].legend([plt.Line2D([0], [0], color='red')],['"truth"'])

    for j, n_obs in enumerate(n_obs_list):
        sns.violinplot(data=data[j,:,:], ax=ax[j])
        for i in range(dim_in):
            ax[j].set_ylabel('%d obs.' % n_obs)
            if data_true is not None:
                ax[j].plot(np.array([i-.25,i+.25]),np.array([data_true[i],data_true[i]]),'red')
        
    ax[0].set_xlim(-.5,dim_in-.5)
    return fig, ax


## miscellaneous


def arrange_full(start, stop, step): 
    # so, e.g., np.arrange(1,10,1) returns [1,2,...,10] instead of [1,2,...,9]
    return np.arange(start, stop+((stop-step)%stop==0), step) 

