import GPy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class GPyVarImportance(object):
    def __init__(self, X, Y, sig2, fix_sig2=True, lengthscale=1.0, variance=1.0):
        super().__init__()

        self.dim_in = X.shape[1]
        self.kernel = GPy.kern.RBF(input_dim=self.dim_in)
        self.model = GPy.models.GPRegression(X,Y,self.kernel)
        self.model.Gaussian_noise.variance = sig2
        
        if fix_sig2:
            self.model.Gaussian_noise.fix()
        
    def train(self):
        self.model.optimize_restarts(num_restarts = 10, verbose=False)
    
    def estimate_psi(self, X, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''
        
        grad_mu, grad_var = self.model.predict_jacobian(X, full_cov=True) # mean and variance of derivative
        #psi = np.mean(grad_mu[:,:,0]**2, axis=0) 
        
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)
        
        for l in range(self.dim_in):
            grad_samp = np.random.multivariate_normal(grad_mu[:,l,0], grad_var[:,:,l,l], size=n_samp) # n_samp_psi x N*
            psi_samp = np.mean(grad_samp**2,1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)
            
        return psi_mean, psi_var
