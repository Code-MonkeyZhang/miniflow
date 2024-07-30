from .Layer_class import Layer
from ..util import*
import numpy as np


class Conv2D(Layer):
    def __init__(self, num_filter, kernel_size, activation, input_shape, stride=(1, 1), padding='valid',
                 layer_name='layer'):

        # Input validation
        # Check if num_filter is a positive integer
        if not isinstance(num_filter, int) or num_filter <= 0:
            raise ValueError("num_filter must be a positive integer")
        # Check if kernel_size is a tuple of two positive integers
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2 and
                all(isinstance(k, int) and k > 0 for k in kernel_size)):
            raise ValueError("kernel_size must be a tuple of two positive integers")
        # Check if input_shape is a tuple of three positive integers
        if not (isinstance(stride, tuple) and len(stride) == 2 and
                all(isinstance(s, int) and s > 0 for s in stride)):
            raise ValueError("stride must be a tuple of two positive integers")
        # Check if padding is either 'valid' or 'same'
        if padding not in ['valid', 'same']:
            raise ValueError("padding must be either 'valid' or 'same'")

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.stride = stride
        self.padding = padding
        self.layer_name = layer_name

        # init weights and biases
        self.Weights = np.zeros((num_filter, kernel_size[0], kernel_size[1], input_shape[2]))
        self.Biases = np.zeros((num_filter, 1))

    def compute_layer(self, A_prev: np.ndarray) -> np.ndarray:

        # Retrieve parameters
        (num_example, example_height, example_width, num_channel) = A_prev.shape
        (num_filter, f_height, f_width, num_channel) = self.Weights.shape

        # Apply Padding
        if self.padding == "valid":
            pad = 0
            A_prev_pad = A_prev
        if self.padding == "same":
            pad = int((f_height - 1) / 2)  # Padding for same
            A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant',
                                constant_values=0)  # pad with zeros

        # Compute the dimensions of the CONV output volume
        # Formula: (n_H - f_H + 2*padding) / stride + 1
        out_height = int((example_height - f_height + 2 * pad) / self.stride[0]) + 1
        out_width = int((example_width - f_width + 2 * pad) / self.stride[1]) + 1

        # Initialize the output volume Z with zeros
        Z = np.zeros((num_example, out_height, out_width, num_filter))

        # Start Convolution
        for i in range(num_example):
            image = A_prev_pad[i]
            for height in range(out_height):
                vert_start = height * self.stride[0]
                vert_end = vert_start + f_height

                for width in range(out_width):
                    horiz_start = width * self.stride[1]
                    horiz_end = horiz_start + f_width

                    for f in range(num_filter):
                        # extract the slice from image， choose all channels
                        # For example, a 6*6*3 image will be sliced to 3*3*3
                        conv_slice = image[vert_start:vert_end, horiz_start:horiz_end, :]

                        # assign the value to Z[i, height, width, f]
                        Z[i, height, width, f] = conv_single_step(conv_slice, self.Weights)

        # Add bias to Z
        Z += self.Biases.reshape(1, 1, 1, num_filter)
        # Apply activation function to the entire Z tensor
        if self.activation:
            Z = self.activation(Z)

        return Z

    def set_weights(self, weights, biases):
        if weights.shape != self.Weights.shape:
            raise ValueError(f"Weights shape mismatch. Expected {self.Weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
            raise ValueError(f"Biases shape mismatch. Expected {self.biases.shape}, got {biases.shape}")


