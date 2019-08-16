#!/usr/bin/env python
# coding: utf-8

# In[159]:


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# generate beta
n = 9
p = 6
beta = tf.random.normal(shape = (p, 1))
g = tf.math.sqrt(tf.matmul(tf.transpose(beta), beta))
v = beta / g
w = tf.concat([g, v], axis = 0)


# In[126]:


# define simulation function
def simu_hessian():
    X = tf.svd(tf.random.normal(shape = (n, p)))[1]
    f = tf.matmul(X, beta)
    y = f + tf.random.normal(shape = (n, 1))  
    loss = tf.reduce_mean((y - w[0] * tf.matmul(X, w[1:] / tf.math.sqrt(tf.matmul(tf.transpose(w[1:]), w[1:]))))**2)
    # dl_dv = tf.gradients(loss, v)
    # d2l_dvdg = tf.gradients(dl_dv, g)
    d2l_dvdg = tf.hessians(loss, w)
    return d2l_dvdg
    


# In[157]:


# generate data
n_sim = 1000
lst = simu_hessian()
for i in range(1, n_sim):
    lst = tf.add(lst, simu_hessian())


# In[158]:


with tf.Session() as session:
    print(session.run(lst / n_sim))

#[[[[[ 2.22221971e-01]
#    [ 2.60303635e-03]
#    [-1.62732578e-03]
#    [ 1.76640635e-03]
#    [-2.80681928e-03]
#    [ 2.74677109e-03]
#    [-8.96020210e-04]]]
#
#
#  [[[ 2.60303472e-03]
#    [ 1.16022658e+00]
#    [ 7.47485012e-02]
#    [ 1.57433808e-01]
#    [-2.32183173e-01]
#    [-1.81377977e-01]
#    [-4.17179972e-01]]]
#
#
#  [[[-1.62732799e-03]
#    [ 7.47485012e-02]
#    [ 1.38890111e+00]
#    [-4.69086468e-02]
#    [ 6.91815466e-02]
#    [ 5.40250875e-02]
#    [ 1.24287896e-01]]]
#
#
#  [[[ 1.76640775e-03]
#    [ 1.57433808e-01]
#    [-4.69086617e-02]
#    [ 1.31244373e+00]
#    [ 1.45589754e-01]
#    [ 1.13806374e-01]
#    [ 2.61653304e-01]]]
#
#
#  [[[-2.80683348e-03]
#    [-2.32183173e-01]
#    [ 6.91815391e-02]
#    [ 1.45589754e-01]
#    [ 1.19645345e+00]
#    [-1.67842150e-01]
#    [-3.85881513e-01]]]
#
#
#  [[[ 2.74677295e-03]
#    [-1.81377992e-01]
#    [ 5.40250950e-02]
#    [ 1.13806352e-01]
#    [-1.67842150e-01]
#    [ 1.28006697e+00]
#    [-3.01558107e-01]]]
#
#
#  [[[-8.96033016e-04]
#    [-4.17179972e-01]
#    [ 1.24287911e-01]
#    [ 2.61653334e-01]
#    [-3.85881454e-01]
#    [-3.01558107e-01]
#    [ 7.17728138e-01]]]]]

