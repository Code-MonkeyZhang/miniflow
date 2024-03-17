import numpy as np
from src.activation import *
from src.util import *


class Layer:
    """
    layer Class
    """

    def __init__(self, units: int, activation: str, name='layer'):
        # init variables
        self.units = units
        self.activation = activation
        self.name = name
        self.Weights = None
        self.Biases = None

    def compute_layer(self, a_in: np.ndarray) -> np.ndarray:
        # init variables
        # iterate and compute each node
        a = np.dot(a_in, self.Weights) + self.Biases
        if self.activation == "sigmoid":
            a_out = sigmoid_function(a_in)
        if self.activation == "linear":
            pass
        return a

    @staticmethod
    def sigmoid_function(z):
        g = 1 / (1 + np.exp(-z))
        return g

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        self.Weights = w
        self.Biases = b
