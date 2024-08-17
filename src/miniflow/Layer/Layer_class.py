from ..activation import *
from ..util import *
import numpy as np


class Layer:
    def __init__(self, layer_name='layer'):
        self.layer_name = layer_name

    def get_output_shape(self, *args, **kwargs):
        pass

    def compute_layer(self, *args, **kwargs):
        pass

    def backward_prop(self, *args, **kwargs):
        pass

    def count_params(self):
        # 初始化参数计数为0
        total_params = 0

        # 如果层有Weights属性，计算权重参数的数量
        if hasattr(self, 'Weights'):
            weight_params = np.prod(self.Weights.shape)
            total_params += weight_params

        # 如果层有Biases属性，计算偏置参数的数量
        if hasattr(self, 'Biases'):
            bias_params = np.prod(self.Biases.shape)
            total_params += bias_params

        # 返回总参数数量
        return total_params
