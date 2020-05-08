import GPy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_in', type=str, default='output/')
parser.add_argument('--dir_out', type=str, default='output/')

args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

res = np.load(os.path.join(args.dir_out, 'results.npy'), allow_pickle=True).item()

# average over reps
psi_mean_mean = np.mean(res['psi_mean'], 2) # mean over psi samples, mean over reps
psi_mean_med = np.median(res['psi_mean'], 2) # mean over psi samples, median over reps

psi_var_mean = np.mean(res['psi_var'], 2) # variance over psi samples, mean over reps
psi_var_med = np.median(res['psi_var'], 2) # variance over psi samples, median over reps


## grid plots
fig, ax = util.plot_results_grid(psi_mean_mean, res['dim_in_list'], res['n_obs_list'])
fig.savefig(os.path.join(args.dir_out, 'estimated_psi.png'))


## violin plots
for idx_dim_in, dim_in in enumerate(res['dim_in_list']):

    fig, ax = util.plot_results_dist(np.squeeze(res['psi_mean'][:,idx_dim_in,:,:]), dim_in, res['n_obs_list'], data_true=None)
    ax[0].set_title('distribution of variable importance $\psi$ \n(rbf_opt input, %d variables)' % dim_in)
    fig.savefig(os.path.join(args.dir_out, 'psi_dist_dim_in=%d.png' % dim_in))




