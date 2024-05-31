from .activation import *
import numpy as np


class Layer:
    def __init__(self, units: int, activation: str, layer_name='layer', input_shape: int = 0):
        self.units = units
        self.activation = activation
        self.layer_name = layer_name
        self.Weights = np.zeros((units, input_shape))
        self.Biases = np.zeros(units)
        self.input_shape = input_shape

    def compute_layer(self, a_in: np.ndarray) -> np.ndarray:
        z = np.dot(a_in, self.Weights.T) + self.Biases
        if self.activation == "sigmoid":
            a_out = sigmoid_function(z)
        if self.activation == "linear":
            a_out = z
        if self.activation == "relu":
            a_out = relu_function(z)
        if self.activation == 'softmax':
            z_max = np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z - z_max)  # 减去z中的最大值以避免溢出
            a_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 按行计算softmax
        return a_out

    def train_layer(self, prev_layer_output, curr_layer_output, label, learningRate,
                    backprop_gradient) -> np.ndarray:

        # 对于最后一层 softmax，cost function的求导就是标签相减
        # 这个计算以后要独立出来，目前先放在这里
        cost_func_gradient = np.subtract(curr_layer_output, label)

        # obtain gradients of weights and bias for updates
        dL_dw, dj_db, dL_dz = self.compute_gradient(prev_layer_output, cost_func_gradient, backprop_gradient)

        # 根据 chain rule, 传给上一层的gradient应该是:
        # dl/da = dl/ds * ds/da
        backprop_gradient = np.dot(dL_dz, self.Weights)

        # perform gradient descent, update gradient
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
            z = np.dot(prev_layer_output, self.Weights.T) + self.Biases
            relu_output = np.maximum(0, z)
            relu_derivative = (relu_output > 0).astype(float)

            dz_dw = prev_layer_output
            dL_dz = backprop_gradient * relu_derivative
            dL_dw = np.dot(dL_dz.T, dz_dw)
            dL_db = np.mean(dL_dz)

        return dL_dw, dL_db, dL_dz

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
    def __init__(self, input_shape, layer_name='Flatten'):
        # Flatten layer doesn't need units & activation
        super().__init__(units=0, layer_name=layer_name, activation="Flatten")
        self.input_shape = input_shape
        self.activation = "Flatten"

    def compute_layer(self, input_array):
        """
         Flattens each element a 1D array.
         Example:
         If input_array has a shape of (1, 28, 28), it will be reshaped to (1,784).
         """

        num_elements = np.prod(input_array.shape[1:])
        output_array = input_array.reshape(
            (input_array.shape[0], num_elements))
        return output_array

    def set_random_weights(self):
        pass
