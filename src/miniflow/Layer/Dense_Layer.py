from ..activation import *
from ..util import *
import numpy as np
from .Layer_class import Layer


class Dense(Layer):
    def __init__(self, units: int, activation: str, input_shape, layer_name='Dense'):
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
        elif self.activation == "linear":
            a_out = z
        elif self.activation == "relu":
            a_out = relu_function(z)
        elif self.activation == 'softmax':
            z_max = np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z - z_max)  # 减去z中的最大值以避免溢出
            a_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 按行计算softmax
        else:
            raise ValueError(
                f"Unsupported activation function: {self.activation}")
        return a_out

    def backward_prop(self, prev_layer_output, prediction, label, alpha, b1, b2, epsilon,
                      backprop_gradient, iter_num) -> np.ndarray:

        # 对于最后一层 softmax，cost function的求导就是标签相减
        # 这个计算以后要独立出来，目前先放在这里
        cost_func_gradient = np.subtract(prediction, label)

        # obtain gradients of weights and bias for updates
        dl_dw, dj_db, dl_dz = self.compute_gradient(
            prev_layer_output, cost_func_gradient, backprop_gradient)

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
        self.Squared_Weights = b2 * self.Squared_Weights + \
            (1 - b2) * np.square(vdw_corrected)
        self.Squared_Biases = b2 * self.Squared_Biases + \
            (1 - b2) * np.square(vdb_corrected)

        sdw_corrected = self.Squared_Weights / (1 - b2 ** iter_num)
        sdb_corrected = self.Squared_Biases / (1 - b2 ** iter_num)

        # perform gradient descent, update gradient
        self.Weights -= alpha * vdw_corrected / \
            np.sqrt(sdw_corrected + epsilon)
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
