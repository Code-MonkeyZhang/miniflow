from ..activation import *
from ..util import *
import numpy as np


class Layer:
    def __init__(self, layer_name='layer'):
        self.layer_name = layer_name

    def compute_layer(self, *args, **kwargs):
        pass
    def forward_prop(self, *args, **kwargs):
        pass
    def compute_gradient(self, *args, **kwargs):
        pass
    def compute_cost_gradient(self, *args, **kwargs):
        pass
    def set_weights(self, *args, **kwargs):
        pass
    def get_weights(self, *args, **kwargs):
        pass
    def get_bias(self, *args, **kwargs):
        pass
    def set_random_weights(self, *args, **kwargs):
        pass


class Dense(Layer):
    def __init__(self, units: int, activation: str, layer_name='layer', input_shape: int = 0):
        self.units = units
        self.activation = activation
        self.layer_name = layer_name
        self.input_shape = input_shape

        self.Weights = np.zeros((units, input_shape))
        self.Biases = np.zeros(units)

        self.Weights_Velocity = np.zeros(self.Weights.shape)
        self.Biases_Velocity = np.zeros(self.Biases.shape)

        self.Squared_Weights = np.zeros(self.Weights.shape)
        self.Squared_Biases = np.zeros(self.Biases.shape)

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

    def forward_prop(self, prev_layer_output, curr_layer_output, label, alpha, b1, b2, epsilon,
                    backprop_gradient, iter_num) -> np.ndarray:

        # 对于最后一层 softmax，cost function的求导就是标签相减
        # 这个计算以后要独立出来，目前先放在这里
        cost_func_gradient = np.subtract(curr_layer_output, label)

        # obtain gradients of weights and bias for updates
        dl_dw, dj_db, dl_dz = self.compute_gradient(prev_layer_output, cost_func_gradient, backprop_gradient)

        # 根据 chain rule, 传给上一层的gradient应该是:
        # dl/da = dl/ds * ds/da
        backprop_gradient = np.dot(dl_dz, self.Weights)

        # GDM
        self.Weights_Velocity = b1 * self.Weights_Velocity + (1 - b1) * dl_dw
        self.Biases_Velocity = b1 * self.Biases_Velocity + (1 - b1) * dj_db

        # Bias Correction for Momentum
        vdw_corrected = self.Weights_Velocity / (1 - b1 ** iter_num)
        vdb_corrected = self.Biases_Velocity / (1 - b1 ** iter_num)

        # RMS prop
        self.Squared_Weights = b2 * self.Squared_Weights + (1 - b2) * np.square(vdw_corrected)
        self.Squared_Biases = b2 * self.Squared_Biases + (1 - b2) * np.square(vdb_corrected)

        sdw_corrected = self.Squared_Weights / (1 - b2 ** iter_num)
        sdb_corrected = self.Squared_Biases / (1 - b2 ** iter_num)

        # perform gradient descent, update gradient
        self.Weights -= alpha * vdw_corrected / np.sqrt(sdw_corrected + epsilon)
        self.Biases -= alpha * vdb_corrected / np.sqrt(sdb_corrected + epsilon)

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

    def get_bias(self) -> np.ndarray:
        return self.Biases

    def set_random_weights(self):
        self.Weights = np.random.randn(*self.Weights.shape)
        self.Biases = np.random.randn(*self.Biases.shape)

    def set_he_weights(self):
        # He初始化权重
        n = self.Weights.shape[1]  # 输入神经元数量
        scale = np.sqrt(2. / n)
        self.Weights = np.random.randn(*self.Weights.shape) * scale

        # 偏置通常初始化为0或很小的常数
        self.Biases = np.zeros(self.Biases.shape)

    def count_params(self):
        # 计算权重参数的数量
        weight_params = np.prod(self.Weights.shape)

        # 计算偏置参数的数量
        bias_params = np.prod(self.Biases.shape)

        # 返回总参数数量
        return weight_params + bias_params

class FlattenLayer(Dense):
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

    def count_params(self):
        return 0

    def output_shape(self):
        return (None, np.prod(self.input_shape))
