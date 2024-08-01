from .Layer_class import Layer
from ..util import *
from ..activation import relu_function
import numpy as np


class MaxPooling2D(Layer):
    def __init__(self, pool_size, input_shape, layer_name='MaxPooling2D'):
        super().__init__(layer_name)
        self.input_shape = input_shape
