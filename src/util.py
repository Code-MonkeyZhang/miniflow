import numpy as np
import copy
import math

# Function to calculate the cost


# Function to calculate the cost
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x_train, y_train, w, b):
    # m = number of training examples
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        # compute the prediction of current w,b
        y_predict = w*x_train[i]+b
        dj_dw_i = (y_predict-y_train[i])*x_train[i]
        dj_db_i = (y_predict-y_train[i])

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

    J_history = []
    w_history = []

    # initialize w and b
    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init)

    # compute gradient

    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        # update w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        print('iteration {}: w = {}, b = {}'.format(i, w, b))
    return w, b
