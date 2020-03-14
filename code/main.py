import gpflow
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from exp.toy import rbf_toy
import exp.util as util
import exp.kernelized as kernel_layers

dataset = "rbf"
fixed_standarization=True
opt_kernel_hyperparam=False
opt_likelihood_variance=False
## data
dataset = dict(rbf=rbf_toy)[dataset]()
seed = 0
n_test = 50
EPOCHS = 32
batch_size = 8
data_dim = 1
rff_dim = 1200
learning_rate = .01

n_train_all = np.arange(40, 210, 20)
n_rep = 10


## allocate space
n_all = n_train_all.size

# summary statistics for GP
rmse = np.zeros((n_all, n_rep))
picp = np.zeros((n_all, n_rep))
mpiw = np.zeros((n_all, n_rep))
test_log_likelihood = np.zeros((n_all, n_rep))

# summary statistics for woodbury-version covariance
rmse2 = np.zeros((n_all, n_rep))
picp2 = np.zeros((n_all, n_rep))
mpiw2 = np.zeros((n_all, n_rep))
test_log_likelihood2 = np.zeros((n_all, n_rep))

# summary statistics for population-version covariance
rmse3 = np.zeros((n_all, n_rep))
picp3 = np.zeros((n_all, n_rep))
mpiw3 = np.zeros((n_all, n_rep))
test_log_likelihood3 = np.zeros((n_all, n_rep))

output_dir = 'output_alt/' + dataset.name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## kernel
if dataset.name == 'rbf':
    kern = gpflow.kernels.RBF(input_dim=1, lengthscales=0.4, variance=1.0)
    dataset.sample_f(n_train_max=np.maximum(1000, np.max(n_train_all)), n_test=n_test, seed=seed)
    # initialize new function

elif dataset.name == 'x3':
    kern = gpflow.kernels.Polynomial(input_dim=1, degree=3, variance=1.0, offset=0.0)

elif dataset.name == 'sin':
    #kern = gpflow.kernels.Periodic(input_dim=1) + gpflow.kernels.RBF(input_dim=1)
    kern = gpflow.kernels.Periodic(input_dim=1, lengthscales=0.002, variance=2.2, period=273.4) + \
           gpflow.kernels.RBF(input_dim=1, lengthscales=31.6, variance=1.0)

elif dataset.name == 'comp':
    #kern = gpflow.kernels.Polynomial(degree=1, input_dim=1) * gpflow.kernels.Periodic(input_dim=1)
    kern = gpflow.kernels.Polynomial(degree=1, input_dim=1, offset=1.11, variance=1.58) * \
           gpflow.kernels.Periodic(input_dim=1, lengthscales=1.20, variance=1.33, period=1.31)


if fixed_standarization:
    x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
    mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
    mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)


for i, n_train in enumerate(n_train_all):

    for j in range(n_rep):
        seed += 2

        original_x_train, original_y_train = \
            dataset.train_samples(n_data=n_train, seed=seed)
        original_x_test, original_y_test = \
            dataset.test_samples(n_data=n_test, seed=seed + 1)

        # standarize data
        if fixed_standarization:
            train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                     original_x_test, original_y_test,
                                                                     mean_x_standard, std_x_standard,
                                                                     mean_y_standard, std_y_standard)
            noise_std = dataset.y_std / std_y_standard

        else:
            train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                     original_x_test, original_y_test)
            noise_std = dataset.y_std / np.std(original_y_train)

        plt.plot(test_x, test_y)
        plt.savefig('temp.png')

        m = gpflow.models.GPR(train_x, train_y, kern=kern)

        # likelihood variance
        if not opt_likelihood_variance:
            m.likelihood.variance.trainable = False
            m.likelihood.variance = noise_std**2 # fixed observation variance

        # optimize hyperparameters
        if opt_kernel_hyperparam:
            #opt = gpflow.train.ScipyOptimizer() # Replace with AdamOptimizer?
            #opt.minimize(m)
            opt = gpflow.train.AdamOptimizer(.05) # Replace with AdamOptimizer?
            opt.minimize(m)

        noise_std_opt = m.likelihood.variance.read_value().item()
        print('noise: ', noise_std_opt)

        print(m.kern.as_pandas_table())

        ## posterior

        ## plot and compute summary statistics using GP
        xx = np.linspace(-2, 2, n_test).reshape(n_test, 1)
        mean, var = m.predict_y(xx)
        samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        #plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:,0],
                         mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                         mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3,3)
        plt.savefig(output_dir + '/gp_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        f_pred, f_pred_cov = m.predict_f_full_cov(test_x)
        f_pred_cov = f_pred_cov[0]
        f_pred_var = np.diagonal(f_pred_cov).reshape(-1, 1)

        f_pred_lb = f_pred - 1.96 * np.sqrt(f_pred_var)
        f_pred_ub = f_pred + 1.96 * np.sqrt(f_pred_var)

        # predictive (i.e. including noise variance)
        y_pred_cov = f_pred_cov + noise_std_opt * np.eye(f_pred.shape[0])
        y_pred, y_pred_var = m.predict_y(test_x)  # y_pred = f_pred

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)
        re_y_pred_cov = y_pred_cov

        ## evaluate posterior
        # y_pred_cov = np.diag(np.diagonal(y_pred_cov)) # TEMP use diagonal of covariance
        test_log_likelihood[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        rmse[i, j] = util.rmse(test_y, y_pred)
        picp[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw[i, j] = util.mpiw(y_pred_lb, y_pred_ub)

        # extract rff features

        model_graph = tf.Graph()

        model_sess = tf.Session(graph=model_graph)

        with model_graph.as_default():
            X = tf.placeholder(dtype=tf.float32, shape=[None, data_dim])

            Y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            H_inv = tf.placeholder(dtype=tf.float32, shape=[rff_dim, rff_dim])

            global_step = tf.Variable(0, trainable=False)

            rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,

                                                            kernel_initializer='gaussian',

                                                            scale=0.4)

            dense_layer = tf.keras.layers.Dense(units=1, activation=None)

            ## define model
            rff_output = rff_layer(X) * np.sqrt(2. / rff_dim)
            Y_pred = dense_layer(rff_output)

            ## define loss
            mean_loss = tf.losses.mean_squared_error(Y_true, Y_pred)

            ## define training
            train_mean = tf.train.AdamOptimizer(learning_rate
                                                ).minimize(mean_loss,
                                                           global_step=global_step)

            tf.summary.scalar("loss", mean_loss)
            tf.summary.histogram("histogram loss", mean_loss)
            summary_op = tf.summary.merge_all()

            weight_cov = util.minibatch_woodbury_update(rff_output, H_inv)

            pred_cov = tf.matmul(rff_output, tf.matmul(weight_cov,
                                                       rff_output, transpose_b=True))


        ### Training and Evaluation ###

        num_batch = int(n_train / batch_size)

        num_steps = num_batch * EPOCHS

        X_batches = np.array_split(train_x[:num_batch * batch_size, ], num_batch) * EPOCHS

        Y_batches = np.array_split(train_y[:num_batch * batch_size, ], num_batch) * EPOCHS


        with model_sess as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('./graphs', model_graph)

            rff_1 = sess.run(rff_output, feed_dict={X: X_batches[0]})

            weight_cov_val = util.compute_inverse(rff_1, sig_sq=noise_std**2)

            for batch_id in range(1, num_steps):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                _, weight_cov_val, rff_output_val, summary = \
                    sess.run([train_mean, weight_cov, rff_output, summary_op],
                             feed_dict={X: X_batch,
                                        Y_true: Y_batch,
                                        H_inv: weight_cov_val})

                writer.add_summary(summary, global_step=batch_id)

            weight_cov_val = weight_cov_val * noise_std**2

            ## preditcion using woodbury-version covariance
            pred_mean_xx, pre_cov_xx = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: xx, Y_true: test_y,
                                    H_inv: weight_cov_val})

            pred_mean_tst, pre_cov_tst = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: test_x,
                                    Y_true: test_y,
                                    H_inv: weight_cov_val})

            ## compute rff, to compute population-version covariance later
            rff_output_val = sess.run(rff_output, feed_dict={X: xx,
                                                             Y_true: test_y,
                                                             H_inv: weight_cov_val})
            train_x_val = sess.run(rff_output, feed_dict={X: train_x,
                                                          Y_true: test_y,
                                                          H_inv: weight_cov_val})
            tst_xx_val = sess.run(rff_output, feed_dict={X: test_x,
                                                         Y_true: test_y,
                                                         H_inv: weight_cov_val})


        ## plot and compute summary statistics using woodbury-version covariance
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions

        f_pred = pred_mean_tst
        f_pred_var = np.diagonal(pre_cov_tst).reshape(-1, 1)

        f_pred_lb = f_pred - 1.96 * np.sqrt(f_pred_var)
        f_pred_ub = f_pred + 1.96 * np.sqrt(f_pred_var)

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)
        ## evaluate posterior
        # y_pred_cov = np.diag(np.diagonal(y_pred_cov)) # TEMP use diagonal of covariance
        test_log_likelihood2[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)

        rmse2[i, j] = util.rmse(test_y, y_pred)
        picp2[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw2[i, j] = util.mpiw(y_pred_lb, y_pred_ub)


        ## plot and compute summary statistics using population-version covariance
        cov_xx_marg = np.matmul(rff_output_val, rff_output_val.T)
        covl = np.matmul(rff_output_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + np.identity(n_train))
        pre_cov_xx = cov_xx_marg - np.matmul(covl, np.matmul(train_x_inv, covl.T))
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)

        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff-pop_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        cov_xx_marg3 = np.matmul(tst_xx_val, tst_xx_val.T)
        covl3 = np.matmul(tst_xx_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + np.identity(n_train))
        pre_cov_tst = cov_xx_marg3 - np.matmul(covl3, np.matmul(train_x_inv, covl3.T))

        f_pred = pred_mean_tst
        f_pred_var = np.diagonal(pre_cov_tst).reshape(-1, 1)

        f_pred_lb = f_pred - 1.96 * np.sqrt(f_pred_var)
        f_pred_ub = f_pred + 1.96 * np.sqrt(f_pred_var)

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)
        ## evaluate posterior
        # y_pred_cov = np.diag(np.diagonal(y_pred_cov)) # TEMP use diagonal of covariance
        test_log_likelihood3[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)

        rmse3[i, j] = util.rmse(test_y, y_pred)
        picp3[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw3[i, j] = util.mpiw(y_pred_lb, y_pred_ub)



## plot

lll_total = np.concatenate([test_log_likelihood, test_log_likelihood2,
                            test_log_likelihood3], axis=0)
rmse_total = np.concatenate([rmse, rmse2, rmse3], axis=0)
picp_total = np.concatenate([picp, picp2, picp3], axis=0)
mpiw_total = np.concatenate([mpiw, mpiw2, mpiw3], axis=0)
lll_min, lll_max = np.min(lll_total), np.max(lll_total)
rmse_min, rmse_max = np.min(rmse_total), np.max(rmse_total)
picp_min, picp_max = np.min(picp_total), np.max(picp_total)
mpiw_min, mpiw_max = np.min(mpiw_total), np.max(mpiw_total)

fig, ax = plt.subplots(1,4,figsize=(16, 4))

ax[0].set_title('log likelihood')
#ax[0].errorbar(n_train_all, np.mean(test_log_likelihood, 1), np.std(test_log_likelihood, 1), None, '-o')
ax[0].boxplot(test_log_likelihood.T, positions=n_train_all)
ax[0].plot(n_train_all, np.mean(test_log_likelihood, 1))
ax[0].set_ylim((lll_min, lll_max))

ax[1].set_title('rmse')
#ax[1].errorbar(n_train_all, np.mean(rmse, 1), np.std(rmse, 1), None, '-o')
ax[1].boxplot(rmse.T, positions=n_train_all)
ax[1].plot(n_train_all, np.mean(rmse, 1))
ax[1].set_ylim((rmse_min, rmse_max))

ax[2].set_title('prediction interval coverage (PICP)')
#ax[2].errorbar(n_train_all, np.mean(picp, 1), np.std(picp, 1), None, '-o')
ax[2].boxplot(picp.T, positions=n_train_all)
ax[2].plot(n_train_all, np.mean(picp, 1))
ax[2].set_ylim((picp_min, picp_max))
ax[2].axhline(y=0.95, color='r', linestyle='dashed')

ax[3].set_title('mean prediction interval width (MPIW)')
#ax[3].errorbar(n_train_all, np.mean(mpiw, 1), np.std(mpiw, 1), None, '-o')
ax[3].boxplot(mpiw.T, positions=n_train_all)
ax[3].plot(n_train_all, np.mean(mpiw, 1))
ax[3].set_ylim((mpiw_min, mpiw_max))

fig.savefig(output_dir + '/results.png')
plt.close()


## plot for rff, woodbury
fig, ax = plt.subplots(1,4,figsize=(16, 4))

ax[0].set_title('log likelihood')
#ax[0].errorbar(n_train_all, np.mean(test_log_likelihood, 1), np.std(test_log_likelihood, 1), None, '-o')
ax[0].boxplot(test_log_likelihood2.T, positions=n_train_all)
ax[0].plot(n_train_all, np.mean(test_log_likelihood2, 1))
ax[0].set_ylim((lll_min, lll_max))

ax[1].set_title('rmse')
#ax[1].errorbar(n_train_all, np.mean(rmse, 1), np.std(rmse, 1), None, '-o')
ax[1].boxplot(rmse2.T, positions=n_train_all)
ax[1].plot(n_train_all, np.mean(rmse2, 1))
ax[1].set_ylim((rmse_min, rmse_max))

ax[2].set_title('prediction interval coverage (PICP)')
#ax[2].errorbar(n_train_all, np.mean(picp, 1), np.std(picp, 1), None, '-o')
ax[2].boxplot(picp2.T, positions=n_train_all)
ax[2].plot(n_train_all, np.mean(picp2, 1))
ax[2].set_ylim((picp_min, picp_max))
ax[2].axhline(y=0.95, color='r', linestyle='dashed')

ax[3].set_title('mean prediction interval width (MPIW)')
#ax[3].errorbar(n_train_all, np.mean(mpiw, 1), np.std(mpiw, 1), None, '-o')
ax[3].boxplot(mpiw2.T, positions=n_train_all)
ax[3].plot(n_train_all, np.mean(mpiw2, 1))
ax[3].set_ylim((mpiw_min, mpiw_max))

fig.savefig(output_dir + '/results2.png')
plt.close()


## plot for rff, population
fig, ax = plt.subplots(1,4,figsize=(16, 4))

ax[0].set_title('log likelihood')
#ax[0].errorbar(n_train_all, np.mean(test_log_likelihood, 1), np.std(test_log_likelihood, 1), None, '-o')
ax[0].boxplot(test_log_likelihood3.T, positions=n_train_all)
ax[0].plot(n_train_all, np.mean(test_log_likelihood3, 1))
ax[0].set_ylim((lll_min, lll_max))

ax[1].set_title('rmse')
#ax[1].errorbar(n_train_all, np.mean(rmse, 1), np.std(rmse, 1), None, '-o')
ax[1].boxplot(rmse3.T, positions=n_train_all)
ax[1].plot(n_train_all, np.mean(rmse3, 1))
ax[1].set_ylim((rmse_min, rmse_max))

ax[2].set_title('prediction interval coverage (PICP)')
#ax[2].errorbar(n_train_all, np.mean(picp, 1), np.std(picp, 1), None, '-o')
ax[2].boxplot(picp3.T, positions=n_train_all)
ax[2].plot(n_train_all, np.mean(picp3, 1))
ax[2].set_ylim((picp_min, picp_max))
ax[2].axhline(y=0.95, color='r', linestyle='dashed')

ax[3].set_title('mean prediction interval width (MPIW)')
#ax[3].errorbar(n_train_all, np.mean(mpiw, 1), np.std(mpiw, 1), None, '-o')
ax[3].boxplot(mpiw3.T, positions=n_train_all)
ax[3].plot(n_train_all, np.mean(mpiw3, 1))
ax[3].set_ylim((mpiw_min, mpiw_max))

fig.savefig(output_dir + '/results3.png')
plt.close()

