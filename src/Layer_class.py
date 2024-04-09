import numpy as np
from src.activation import *
from src.util import *


class Layer:
    """
    layer Class
    """

    def __init__(self, units: int, activation: str, name='layer', input_shape: int = 0):
        # init variables
        self.units = units
        self.activation = activation
        self.name = name
        self.Weights = np.zeros((units, input_shape))
        self.Biases = np.zeros(units)
        self.input_shape = input_shape

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

            z = linear_function(a_in, self.Weights, self.Biases)
            exp_z = np.exp(z)
            a_out = exp_z / np.sum(exp_z)

        return a_out

    """
    This function performs Two major tasks:
    - obtain weights and bias
    - do gradient descent
    """

    def train_layer(self, prev_layer_output, curr_layer_output, label, learningRate,
                    backprop_gradient) -> np.ndarray:
        # 暂且是这样
        cost_func_gradient = np.subtract(curr_layer_output, label)
        # obtain gradients of weights and bias for updates
        dL_dw, dj_db = self.compute_gradient(prev_layer_output, cost_func_gradient, backprop_gradient)

        # 根据 chain rule, 传给上一层的gradient应该是:
        # dl/da = dl/ds * ds/da
        if self.activation == "softmax":
            backprop_gradient = np.dot(cost_func_gradient, self.Weights)

        # do gradient descent, update gradient
        self.Weights -= learningRate * dL_dw
        self.Biases -= learningRate * dj_db

        return backprop_gradient

    def compute_gradient(self, prev_layer_output, cost_func_gradient, backprop_gradient) -> np.ndarray:
        if self.activation == "softmax":
            dL_dz = cost_func_gradient
            # linear 的一阶导是x,也就是这一层的输入
            dz_dw = prev_layer_output
            dL_dw = np.dot(dz_dw.T, dL_dz)
            # multiply gradients with the backprop gradients from the prev layer
            # if this is the last layer, backprop gradients are all 1s
            dL_dw = np.multiply(backprop_gradient, dL_dw.T)

            dL_db = np.mean(dL_dz, axis=0)
        if self.activation == "relu":
            # linear 的一阶导是x,也就是这一层的输入
            dz_dw = prev_layer_output
            # relu_derivative = (dz_dw > 0).astype(float)
            dL_dz = backprop_gradient
            dL_dw = np.dot(dL_dz.T, dz_dw)

            dL_db = np.mean(dL_dz)

        return dL_dw, dL_db

    def compute_cost_gradient(self, prediction, label):
        if self.activation == "softmax":
            cost_gradient = np.subtract(prediction, label)

        return cost_gradient

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        self.Weights = w
        self.Biases = b

    def get_weights(self) -> np.ndarray:
        return self.Weights

    def set_random_weights(self):
        self.Weights = np.random.randn(*self.Weights.shape)
        self.Biases = np.random.randn(*self.Biases.shape)


class FlattenLayer(Layer):
    def __init__(self, input_shape, name='Flatten'):
        # 由于Flatten层不需要units和activation，我们可以传递默认值或None
        super().__init__(units=0, name=name, activation="Flatten")
        self.input_shape = input_shape
        self.activation = "Flatten"

    def compute_layer(self, input_array):
        num_elements = np.prod(input_array.shape[1:])
        output_array = input_array.reshape(
            (input_array.shape[0], num_elements))
        return output_array

    def train_layer(self, prev_layer_output, curr_layer_output, label, learningRate,
                    backprop_gradient) -> np.ndarray:
        pass

    def set_random_weights(self):
        pass
