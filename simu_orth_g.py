import tensorflow as tf
import numpy as np

n = 1000
P = 6
K = 5

# X = tf.Variable(tf.random_normal([n, P]))
# for p in range(P):
#      exec(f'x{p + 1} = tf.transpose(X)[p]')

# define weights
W = tf.Variable(tf.random_normal([P, K]))
beta = tf.Variable(tf.random_normal([K, 1]))

# define network
# h = tf.matmul(X, W)
# a = tf.nn.relu(h)
# f = tf.matmul(a, beta)

# generate data using numpy
# noise = np.random.normal(size=n)
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     f_obs = sess.run(f)
# #     y = f_obs.flatten() + noise
# #     print(f_obs)
# #     print(y)
# #
# # # define loss and score functions
# # loss = tf.reduce_mean(tf.square(y - f))
# # df_dX = tf.transpose(tf.gradients(f, X)[0])
# # psi3 = tf.reduce_mean(tf.square(df_dX[2]))
# # psi4 = tf.reduce_mean(tf.square(df_dX[3]))
# # dl_d3 = tf.gradients(loss, psi3)
# # dl_d4 = tf.gradients(loss, psi4)
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print(sess.run(psi3) * sess.run(psi4))


def simu_orth():
    # X = tf.Variable(tf.random_normal([n, P]))
    X = tf.svd(tf.random.normal(shape=(n, P)))[1]
    # for p in range(P):
    #      exec(f'x{p + 1} = tf.transpose(X)[p]')

    # define network
    h = tf.matmul(X, W)
    a = tf.nn.relu(h)
    f = tf.matmul(a, beta)

    # generate data using numpy
    noise = np.random.normal(size=n)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        f_obs = sess.run(f)
        y = f_obs.flatten() + noise

    # define loss and score functions
    loss = tf.reduce_mean(tf.square(y - f))
    df_dX = tf.transpose(tf.gradients(f, X)[0])
    psi3 = tf.reduce_mean(tf.square(df_dX[2]))
    psi4 = tf.reduce_mean(tf.square(df_dX[3]))
    dl_d3 = tf.gradients(loss, psi3)
    dl_d4 = tf.gradients(loss, psi4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prod = sess.run(psi3) * sess.run(psi4)
    return prod

# generate data
n_sim = 200
lst = simu_orth()
for i in range(1, n_sim):
    lst = lst + simu_orth()

print(lst / n_sim)
# 12.563, X = tf.Variable(tf.random_normal([n, P]))
# 9.682, X = tf.svd(tf.random.normal(shape=(n, P)))[1]
