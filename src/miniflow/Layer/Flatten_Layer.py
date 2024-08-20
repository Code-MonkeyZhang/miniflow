from ..activation import *
from ..util import *
import numpy as np
from .Layer_class import Layer


class FlattenLayer(Layer):
    def __init__(self, input_shape, layer_name='Flatten'):
        # Flatten layer doesn't need units & activation
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape()

    def compute_layer(self, input_array):
        """
         Flattens each element a 1D array.
         Example:
         If input_array has a shape of (1, 28, 28), it will be reshaped to (1,784).
         """
        self.cache = input_array.shape  # Storing input shape in cache for backprop
        num_elements = np.prod(input_array.shape[1:])
        output_array = input_array.reshape(
            (input_array.shape[0], num_elements))
        return output_array

    def set_random_weights(self):
        pass

    def count_params(self):
        return 0

    def get_output_shape(self):
        return np.prod(self.input_shape)

    def backward_prop(self, dA):
        # Retrieve the original input shape from cache
        input_shape = self.cache
        # Reshape the gradient to the original input shape
        dA_prev = dA.reshape(input_shape)
        return dA_prev
