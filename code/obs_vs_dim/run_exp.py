import GPy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import util
import models

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='rbf')
parser.add_argument('--dir_out', type=str, default='output/')

parser.add_argument('--n_obs_min', type=int, default=10)
parser.add_argument('--n_obs_max', type=int, default=100)
parser.add_argument('--n_obs_step', type=int, default=10)

parser.add_argument('--n_obs_true', type=int, default=200)

parser.add_argument('--dim_in_min', type=int, default=2)
parser.add_argument('--dim_in_max', type=int, default=5)
parser.add_argument('--dim_in_step', type=int, default=2)

parser.add_argument('--n_rep', type=int, default=2)

parser.add_argument('--model', type=str, default='GP', help='select "GP" or "RFF"')

# GP options
parser.add_argument('--opt_likelihood_variance', action='store_true')
parser.add_argument('--opt_kernel_hyperparam', action='store_true')
parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
parser.add_argument('--kernel_variance', type=float, default=1.0)

# RFF options
parser.add_argument('--rff_dim', type=int, default=1200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=16)


args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

# allocate space
n_obs_list = util.arrange_full(args.n_obs_min, args.n_obs_max, args.n_obs_step)
dim_in_list = util.arrange_full(args.dim_in_min, args.dim_in_max, args.dim_in_step)

## allocate space for results: obs x dim_in x rep x input
res_shape = (len(n_obs_list), len(dim_in_list), args.n_rep, np.max(dim_in_list))
res = {
    'psi_mean': np.full(res_shape, np.nan),
    'psi_var': np.full(res_shape, np.nan),
    'n_obs_list': n_obs_list,
    'dim_in_list': dim_in_list
    }

seed = 0
for i, n_obs in enumerate(n_obs_list):
    for j, dim_in in enumerate(dim_in_list):
        print('n_obs [%d/%d], dim_in [%d/%d]' % (i,len(n_obs_list),j,len(dim_in_list)))
        
        for k in range(args.n_rep):
            seed += 1

            Z, X, Y, sig2 = util.load_data(args.dataset_name, n_obs=n_obs, dim_in=dim_in, seed=seed)

            if args.model=='GP':
                m = models.GPyVarImportance(Z, Y, sig2=sig2, \
                    opt_kernel_hyperparam=args.opt_kernel_hyperparam, \
                    opt_sig2=args.opt_likelihood_variance,\
                    lengthscale=args.kernel_lengthscale, variance=args.kernel_variance)

                m.train()
            
            elif args.model=='RFF':
                m = models.RffVarImportance(Z)
                m.train(Z, Y, sig2, rff_dim=args.rff_dim, batch_size=args.batch_size, epochs=args.epochs)

            psi_est = m.estimate_psi(Z)
            res['psi_mean'][i,j,k,:dim_in] = psi_est[0]
            res['psi_var'][i,j,k,:dim_in] = psi_est[1]


np.save(os.path.join(args.dir_out, 'results.npy'), res)


                