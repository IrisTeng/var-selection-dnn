import numpy as np
import os
import tensorflow as tf

from exp.toy import rbf_toy
import exp.util as util
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as keras_backend
import exp.kernelized as kernel_layers

dataset = "rbf"
fixed_standarization = True
## data
dataset = dict(rbf=rbf_toy)[dataset]()
# n_train_all = np.arange(10,50,10)
n_train_all = np.arange(100, 220, 50)
# n_train_all = np.array([250])
n_rep = 2
seed = 0
n_test = 1000
rff_K = 10
batch_size = 20

## allocate space
n_all = n_train_all.size

output_dir = 'output_alt/' + dataset.name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## kernel
if dataset.name == 'rbf':
    # kern = gpflow.kernels.RBF(input_dim=1, lengthscales=1.0, variance=1.0)
    dataset.sample_f(n_train_max=np.maximum(1000, np.max(n_train_all)), n_test=n_test, seed=seed)
    # initialize new function

if fixed_standarization:
    x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
    mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
    mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

n_train = 100

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

tst = test_util.TensorFlowTestCase()
rff_layer = kernel_layers.RandomFourierFeatures(
    output_dim=rff_K,
    kernel_initializer='gaussian',
    trainable=True)

model = tf.keras.models.Sequential()
model.add(rff_layer)
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',  # mean squared error
              metrics=['mae'])  # mean absolute error
EPOCHS = 1000
model.fit(train_x, train_y,
          epochs=EPOCHS, validation_split=0.2, verbose=0)
# loss, mae, mse = model.evaluate(test_x, test_y, verbose=2)
# print("Testing set Mean Abs Error: {:5.2f}".format(mae))
batch_round = int(n_train / batch_size) + 1
H_inv = noise_std ** (-2) * tf.eye(rff_K)
for l in range(batch_round):
    X = tf.cast(util.get_Batch(train_x, train_y, batch_size), dtype=tf.float32)
    rff_X = np.sqrt(2.0 / rff_K) * rff_layer(X)
    H_inv = util.minibatch_woodbury_update(rff_X, H_inv)
Sigma_beta = noise_std ** 2 * H_inv

## posterior

# plot
xx = np.linspace(-2, 2, 100).reshape(100, 1)  # test points must be of shape (N, D)
mean = model.predict(xx).flatten()
output_xx = np.sqrt(2.0 / rff_K) * rff_layer(xx)
var = tf.matmul(output_xx, tf.matmul(Sigma_beta, output_xx, transpose_b=True))
#init_op = tf.initializers.global_variables()
with tst.cached_session() as sess:
    keras_backend._initialize_variables(sess)
    #sess.run(init_op)
    var = sess.run(var)

