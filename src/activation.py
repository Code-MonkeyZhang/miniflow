import numpy as np


def linear_function(x_train, w, b):
    return np.dot(x_train, w) + b


def sigmoid_function(z):
    g = 1 / (1 + np.exp(-z))
    return g


def relu_function(z):
    return np.maximum(0, z)
