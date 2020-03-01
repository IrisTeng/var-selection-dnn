import gpflow
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from exp.toy import rbf_toy
import exp.util as util
import exp.kernelized as kernel_layers

dataset = "rbf"
fixed_standarization = True
## data
dataset = dict(rbf=rbf_toy)[dataset]()
seed = 0
n_test = 1000
n_train = 99

output_dir = 'output_alt/' + dataset.name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

kern = gpflow.kernels.RBF(input_dim=1, lengthscales=1.0, variance=1.0)
dataset.sample_f(n_train_max=np.maximum(1000, np.max(1000)), n_test=n_test, seed=seed)
# initialize new function

x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

## data
original_x_train, original_y_train = \
    dataset.train_samples(n_data=n_train, seed=seed)
original_x_test, original_y_test = \
    dataset.test_samples(n_data=n_test, seed=seed + 1)

# standarize data
train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                         original_x_test, original_y_test,
                                                         mean_x_standard, std_x_standard,
                                                         mean_y_standard, std_y_standard)
noise_std = dataset.y_std / std_y_standard
## model for rff
model_graph = tf.Graph()
model_sess = tf.Session(graph=model_graph)

### Define Model ###
batch_size = 10
data_dim = 1
rff_dim = 1000
learning_rate = .01

with model_graph.as_default():
    ## define input
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, data_dim])
    Y_true = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
    H_inv = tf.placeholder(dtype=tf.float32, shape=[rff_dim, rff_dim])
    global_step = tf.Variable(0, trainable=False)

    ## define model layers
    rff_layer = kernel_layers.RandomFourierFeatures(
        output_dim=rff_dim,
        kernel_initializer='gaussian',
        scale=1.0)
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
    ## normalize weight_cov using num_sample=global_step * batch_size???

    pred_cov = tf.matmul(rff_output, tf.matmul(weight_cov,
                                               rff_output, transpose_b=True))
    ## define initialization
    init_op = tf.initialize_all_variables()

    ## model_graph.finalize()

### Training and Evaluation ###
weight_cov_val = noise_std ** (-2) * np.eye(rff_dim)
num_steps = int(n_train / batch_size)
X_batches = np.array_split(train_x[:num_steps * batch_size, ], num_steps)
Y_batches = np.array_split(train_y[:num_steps * batch_size, ], num_steps)

with model_sess as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs', model_graph)

    for batch_id in range(num_steps):
        X_batch = X_batches[batch_id]
        Y_batch = Y_batches[batch_id]

        ## update posterior mean/covariance
        _, weight_cov_val, summary = \
            sess.run([train_mean, weight_cov, summary_op],
                     feed_dict={X: X_batch,
                                Y_true: Y_batch,
                                H_inv: weight_cov_val})

        ## evaluate posterior predictive
        # pred_mean_val, pre_cov_val, summary = \
        #     sess.run([Y_pred, pred_cov, summary_op],
        #              feed_dict={X: test_x,
        #                         Y_true: test_y,
        #                         H_inv: weight_cov_val})
        # do we need to split test dataset into batch_size?

        writer.add_summary(summary, global_step=batch_id)
        ## compute metric, visualization

### no epoch? how to predict?
