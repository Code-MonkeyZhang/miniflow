from .Layer_class import Layer
from ..util import *
from ..activation import relu_function
import numpy as np


class MaxPooling2D(Layer):
    def __init__(
            self,
            pool_size,
            input_shape,
            stride=None,
            layer_name="MaxPooling2D",
    ):
        super().__init__(layer_name)

        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = (
            stride if stride is not None else pool_size
        )  # if stride is not assigned, set stride = pool_size by default

        self.output_shape = self.get_output_shape()

    def compute_layer(self, a_in: np.ndarray) -> np.ndarray:
        """
        This function does two things:
        1. Computes max pooling
        2. generates corresponding mask.

        Applies max pooling to 'a_in' using specified pool size and stride.
        Also generates a mask identifying the positions of max values, for use in backpropagation.

        Args:
            a_in (np.ndarray): Input from the previous layer

        Returns:
            np.ndarray: Pooled output
        """

        # Retrieve variables
        (num_example, in_h, in_w, in_ch) = a_in.shape
        (pool_h, pool_w) = self.pool_size
        (stride_h, stride_w) = self.stride
        (out_h, out_w, out_ch) = self.output_shape

        # initialize output
        Z = np.zeros((num_example, out_h, out_w, in_ch))
        # initialize mask to all False
        self.mask = np.zeros(a_in.shape, dtype=bool)

        for h in range(out_h):
            for w in range(out_w):
                vert_start = h * stride_h
                vert_end = vert_start + pool_h
                horiz_start = w * stride_w
                horiz_end = horiz_start + pool_w

                a_prev_slice = a_in[:, vert_start:vert_end,
                                    horiz_start:horiz_end, :]

                # Calculate max values keeping the dimensions for broadcasting
                max_values = np.max(a_prev_slice, axis=(1, 2), keepdims=True)

                # Create a mask for the max values
                mask = (a_prev_slice == max_values)
                # Assign mask to the corresponding positions in the mask matrix
                self.mask[:, vert_start:vert_end,
                          horiz_start:horiz_end, :] = mask
                # Store the max values in the output matrix
                Z[:, h, w, :] = max_values.squeeze((1, 2))

        return Z

    def get_output_shape(self):
        (in_h, in_w, num_ch) = self.input_shape
        (pool_h, pool_w) = self.pool_size
        (stride_h, stride_w) = self.stride

        # 计算输出高度和宽度
        out_h = (in_h - pool_h) // stride_h + 1
        out_w = (in_w - pool_w) // stride_w + 1

        output_shape = (out_h, out_w, num_ch)
        return output_shape

    def backward_prop(self, dA):
        # Initialize dA_prev as the same shape as the input mask to store the gradients passed back to the input
        dA_prev = np.zeros_like(self.mask, dtype=np.float32)

        # Retrieve dimensions from dA_prev and dA to use in loops
        (num_example, in_h, in_w, in_ch) = dA_prev.shape
        (pool_h, pool_w) = self.pool_size
        (stride_h, stride_w) = self.stride
        (num_example, out_h, out_w, out_ch) = dA.shape

        # Loop over each position in the output gradient matrix
        for h in range(out_h):
            for w in range(out_w):
                # Calculate the start and end indices of the window in the input corresponding to the current output element
                vert_start = h * stride_h
                vert_end = vert_start + pool_h
                horiz_start = w * stride_w
                horiz_end = horiz_start + pool_w

                # Distribute the gradient from the output back to the input only at the positions where the maximum was located# Using the stored mask to apply gradients only where the input had the maximum value
                grad_at_position = dA[:, h, w, :][:, np.newaxis, np.newaxis, :]
                mask_slice = self.mask[:, vert_start:vert_end,
                                       horiz_start:horiz_end, :]
                dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += (
                    grad_at_position *
                    mask_slice)

            # Return the propagated gradient matrix for the input layerreturn dA_prev
        return dA_prev
