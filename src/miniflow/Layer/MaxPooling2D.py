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
        # Retrieve variables
        (num_example, in_h, in_w, in_ch) = a_in.shape
        (pool_h, pool_w) = self.pool_size
        (stride_h, stride_w) = self.stride
        (out_h, out_w, out_ch) = self.output_shape

        # initialize output
        Z = np.zeros((num_example, out_h, out_w, in_ch))

        for h in range(out_h):
            for w in range(out_w):
                vert_start = h * stride_h
                vert_end = vert_start + pool_h
                horiz_start = w * stride_w
                horiz_end = horiz_start + pool_w

                a_prev_slice = a_in[:, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, h, w, :] = np.max(a_prev_slice, axis=(1, 2))

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
