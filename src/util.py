import numpy as np

# compute gradient


def compute_gradient(x_train, y_train, w, b):
    # m = number of training examples
    m = x_train.shape[0]
    dj_dw = 0

    dj_db = 0
    for i in range(m):
        # compute the prediction of current w,b
        y_predict = w*x_train[i]+b
        dj_dw_i = (y_predict-y_train)*x_train[i]
        dj_db_i = (y_predict-y_train)

        # sum all m parameters
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

# feature scaling


def feature_scaling(x_train):
    return x_train


# gradient descent
def simple_linear_regression(x_train, y_train, w_init, b_init, alpha, num_iterations):

    # initialize w and b
    w = w_init
    b = b_init

    # compute gradient
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

    # update w and b
    w = w - alpha * dj_dw
    b = b = alpha * dj_db
    return w, b
