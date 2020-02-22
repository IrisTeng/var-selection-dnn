import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

def standardize_data(original_x_train, original_y_train, original_x_test, original_y_test,
                     mean_x = None, std_x = None, mean_y = None, std_y = None):
    
    if mean_x is not None and std_x is not None:
        mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)

    if mean_y is not None and std_y is not None:
        mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
    
    train_x = (original_x_train - mean_x) / std_x
    train_y = ((original_y_train - mean_y) / std_y).reshape(-1,1)

    test_x = ((original_x_test - mean_x) / std_x).reshape(-1,1)
    test_y = ((original_y_test - mean_y) / std_y).reshape(-1,1)

    return train_x, train_y, test_x, test_y


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


def minibatch_woodbury_update(X, H_inv):
    """Minibatch update of linear regression posterior covariance
    using Woodbury matrix identity.

    inv(H + X^T X) = H_inv - H_inv X^T inv(I + X H_inv X^T) X H_inv

    Args:
        X: (tf.Tensor) A M x K matrix of batched observation.
        H_inv: (tf.Tensor) A K x K matrix of posterior covariance of rff coefficients.


    Returns:
        H_new: (tf.Tensor) K x K covariance matrix after update.
    """
    batch_size, dim_size = X.shape.as_list()

    M0 = tf.eye(batch_size) + tf.matmul(X, tf.matmul(H_inv, X, transpose_b=True))
    M = tf.matrix_inverse(M0)
    B = tf.matmul(X, H_inv)
    H_new = H_inv - tf.matmul(B, tf.matmul(M, B), transpose_a=True)

    return H_new


def get_Batch(data, label, batch_size):
    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=False, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32,
                                      allow_smaller_final_batch=False)
    return x_batch




