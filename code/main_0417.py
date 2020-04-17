import gpflow
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

from exp.toy import rbf_toy
import exp.util as util
import exp.kernelized as kernel_layers

dataset = "rbf"
fixed_standarization = True
opt_kernel_hyperparam = False
opt_likelihood_variance = False
## data
dataset = dict(rbf=rbf_toy)[dataset]()
seed = 0
n_test = 50
EPOCHS = 15
batch_size = 16
data_dim = 1
rff_dim = 10
learning_rate = .01

output_dir = 'output/rff_tst/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## kernel
kern = gpflow.kernels.RBF(input_dim=1, lengthscales=0.4, variance=1.0)
dataset.sample_f(n_train_max=1000, n_test=n_test, seed=seed)

n_train = 1000

x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

original_x_train, original_y_train = \
    dataset.train_samples(n_data=n_train, seed=seed)
original_x_test, original_y_test = \
    dataset.test_samples(n_data=n_test, seed=seed + 1)

train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                         original_x_test, original_y_test,
                                                         mean_x_standard, std_x_standard,
                                                         mean_y_standard, std_y_standard)
noise_std = dataset.y_std / std_y_standard

xx = np.linspace(-2, 2, n_test).reshape(n_test, 1)






# extract rff features

model_graph = tf.Graph()

model_sess = tf.Session(graph=model_graph)

with model_graph.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[None, data_dim])

    Y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    H_inv = tf.placeholder(dtype=tf.float32, shape=[rff_dim, rff_dim])

    Phi_y = tf.placeholder(dtype=tf.float32, shape=[rff_dim, 1])

    global_step = tf.Variable(0, trainable=False)

    rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,

                                                    kernel_initializer='gaussian',

                                                    scale=0.4)

    dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                        kernel_regularizer=keras.regularizers.l2(l=noise_std ** 2))

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

    covl_xy = util.minibatch_interaction_update(Phi_y, rff_output, Y_true)

    pred_cov = tf.matmul(rff_output, tf.matmul(weight_cov,
                                               rff_output, transpose_b=True))

### Training and Evaluation ###

X_batches = util.split_into_batches(train_x, batch_size) * EPOCHS

Y_batches = util.split_into_batches(train_y, batch_size) * EPOCHS

num_steps = X_batches.__len__()

num_batch = int(num_steps / EPOCHS)

with model_sess as sess:
    sess.run(tf.global_variables_initializer())

    writer2 = tf.summary.FileWriter('./graphs', model_graph)

    rff_1 = sess.run(rff_output, feed_dict={X: X_batches[0]})

    weight_cov_val = util.compute_inverse(rff_1, sig_sq=noise_std ** 2)

    covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

    for batch_id in range(1, num_batch):
        X_batch = X_batches[batch_id]
        Y_batch = Y_batches[batch_id]

        ## update posterior mean/covariance
        try:
            weight_cov_val, covl_xy_val = sess.run([weight_cov, covl_xy],
                                                   feed_dict={X: X_batch,
                                                              Y_true: Y_batch,
                                                              H_inv: weight_cov_val,
                                                              Phi_y: covl_xy_val})

        except:
            print("\n================================\n"
                  "Problem occurred at Step {}\n"
                  "================================".format(batch_id))

        weight_tmp = weight_cov_val
        covl_tmp = covl_xy_val

    for batch_id in range(0, num_steps):
        X_batch = X_batches[batch_id]
        Y_batch = Y_batches[batch_id]

        ## update posterior mean/covariance
        try:
            _, summary = sess.run([train_mean, summary_op],
                                  feed_dict={X: X_batch,
                                             Y_true: Y_batch,
                                             H_inv: weight_cov_val,
                                             Phi_y: covl_xy_val})

        except:
            print("\n================================\n"
                  "Problem occurred at Step {}\n"
                  "================================".format(batch_id))

    beta = np.matmul(weight_cov_val, covl_xy_val)

    weight_cov_val = weight_cov_val * noise_std ** 2

    ## prediction using woodbury-version covariance
    pred_mean_xx, pre_cov_xx = \
        sess.run([Y_pred, pred_cov],
                 feed_dict={X: xx, Y_true: test_y,
                            H_inv: weight_cov_val})

    pred_mean_tst, pre_cov_tst = \
        sess.run([Y_pred, pred_cov],
                 feed_dict={X: test_x,
                            Y_true: test_y,
                            H_inv: weight_cov_val})

    # compute rff, to compute population-version covariance later
    rff_output_val = sess.run(rff_output, feed_dict={X: xx,
                                                     Y_true: test_y,
                                                     H_inv: weight_cov_val})
    train_x_val = sess.run(rff_output, feed_dict={X: train_x,
                                                  Y_true: test_y,
                                                  H_inv: weight_cov_val})
    tst_xx_val = sess.run(rff_output, feed_dict={X: test_x,
                                                 Y_true: test_y,
                                                 H_inv: weight_cov_val})



## weight covariance matrix for all data, should equal to weight_tmp
weight_cov_all = util.compute_inverse(train_x_val, sig_sq=noise_std ** 2)
np.max(np.abs(weight_tmp - weight_cov_all))




# plot and compute summary statistics using woodbury-version covariance
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
plt.savefig(output_dir + '/rff_n_train=%d_rep=%d.png' % (n_train, 6))
plt.close()




## plot and compute summary statistics using bayesian-lnr mean prediction

pred_mean_xx = np.matmul(rff_output_val, beta)
plt.figure(figsize=(12, 6))
plt.plot(train_x, train_y, 'kx', mew=2)
# plt.plot(test_x, test_y, 'rx', mew=2)
plt.plot(xx, pred_mean_xx, 'C0', lw=2)
plt.fill_between(xx[:, 0],
                 pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                 pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                 color='C0', alpha=0.2)
plt.ylim(-3, 3)
plt.savefig(output_dir + '/lnr_n_train=%d_rep=%d.png' % (n_train, 6))
plt.close()





## plot and compute summary statistics using population-version covariance
cov_xx_marg = np.matmul(rff_output_val, rff_output_val.T)
covl = np.matmul(rff_output_val, train_x_val.T)
train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                            + noise_std ** 2 * np.identity(n_train))

pred_mean_xx = np.matmul(covl, np.matmul(train_x_inv, train_y))
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
plt.savefig(output_dir + '/rff-pop_n_train=%d_rep=%d.png' % (n_train, 6))
plt.close()

