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
        self.Weights = np.zeros(units)
        self.Biases = np.zeros(units)

    def compute_layer(self, a_in: np.ndarray) -> np.ndarray:
        # init variables
        # iterate and compute each node
        if self.activation == "sigmoid":
            z = np.dot(a_in, self.Weights) + self.Biases
            a_out = sigmoid_function(z)
        if self.activation == "linear":
            a_out = linear_function(a_in, self.Weights, self.Biases)
        if self.activation == "relu":
            # Linear
            z = linear_function(a_in, self.Weights, self.Biases)
            a_out = relu_function(z)
        if self.activation == 'softmax':
            a_out = np.zeros(self.units)
            z = np.zeros(self.units)
            # The first loop compute all inputs for sigmoid
            z = linear_function(a_in, self.Weights, self.Biases)
            exp_z = np.exp(z)
            a_out = exp_z / np.sum(exp_z)

        return a_out

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        self.Weights = w
        self.Biases = b


class FlattenLayer(Layer):
    def __init__(self, input_shape, name='Flatten'):
        # 由于Flatten层不需要units和activation，我们可以传递默认值或None
        super().__init__(units=0, activation=None, name=name)
        self.input_shape = input_shape

    def compute_layer(self, input_array):
        num_elements = np.prod(input_array.shape[1:])
        output_array = input_array.reshape((input_array.shape[0], num_elements))
        return output_array
