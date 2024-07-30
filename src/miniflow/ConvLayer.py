from .Layer_class import Layer
import numpy as np


class Conv2D(Layer):
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


def conv_single_step(a_slice_prev, conv_weights, b):
    """
    Simple Convolution operation for a single step.
    It multiplies a_slice_prev and conv_weights, and then sums over all entries.
    Basic building block for a convolutional layer.
    """
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, conv_weights)
    
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z