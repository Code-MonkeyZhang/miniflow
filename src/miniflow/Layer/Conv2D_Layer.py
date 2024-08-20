from .Layer_class import Layer
from ..util import *
from ..activation import relu_function
import numpy as np


class Conv2D(Layer):
    def __init__(self, num_filter, kernel_size, activation, input_shape, stride=(1, 1), padding='valid',
                 layer_name='Conv2D'):

        # Input validation
        # Check if num_filter is a positive integer
        if not isinstance(num_filter, int) or num_filter <= 0:
            raise ValueError("num_filter must be a positive integer")
        # Check if kernel_size is a tuple of two positive integers
        if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2 and
                all(isinstance(k, int) and k > 0 for k in kernel_size)):
            raise ValueError(
                "kernel_size must be a tuple of two positive integers")
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
        self.Weights = np.zeros(
            (kernel_size[0], kernel_size[1], input_shape[2], num_filter))
        self.Biases = np.zeros((num_filter))

        self.Weights_Velocity = np.zeros(self.Weights.shape)
        self.Biases_Velocity = np.zeros(self.Biases.shape)

        self.Squared_Weights = np.zeros(self.Weights.shape)
        self.Squared_Biases = np.zeros(self.Biases.shape)

    def compute_layer(self, A_prev: np.ndarray) -> np.ndarray:

        # Retrieve parameters
        (num_example, example_height, example_width, in_num_channel) = A_prev.shape
        (f_height, f_width, out_num_channel, num_filter) = self.Weights.shape
        (vert_stride, horiz_stride) = self.stride

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
        out_height = int((example_height - f_height +
                          2 * pad) / vert_stride) + 1
        out_width = int((example_width - f_width + 2 * pad) /
                        horiz_stride) + 1

        # Initialize the output volume Z with zeros
        Z = np.zeros((num_example, out_height, out_width, num_filter))

        # Start Convolution
        for height in range(out_height):

            vert_start = height * vert_stride
            vert_end = vert_start + f_height

            for width in range(out_width):

                horiz_start = width * horiz_stride
                horiz_end = horiz_start + f_width

                for f in range(num_filter):
                    # extract the slice from all images, choose all channels
                    conv_slice = A_prev_pad[:, vert_start:vert_end,
                                            horiz_start:horiz_end, :]
                    # Perform convolution operation & store the result in corresponding position in Z
                    conv_result = np.sum(
                        conv_slice * self.Weights[:, :, :, f], axis=(1, 2, 3))
                    Z[:, height, width, f] = conv_result

        # Add bias to Z
        Z += self.Biases.reshape(1, 1, 1, num_filter)

        # Apply activation function to the entire Z tensor
        if self.activation == 'relu':
            Z = relu_function(Z)
        return Z
        return Z

    def backward_prop(self, prev_layer_output, dA, learning_rate, b1, b2, epsilon, iter_num):
        # 获取维度信息
        (num_example, n_H_prev, n_W_prev, n_C_prev) = prev_layer_output.shape
        (f, f, n_C_prev, n_C) = self.Weights.shape

        # 初始化梯度
        dA_prev = np.zeros_like(prev_layer_output)
        dW = np.zeros_like(self.Weights)
        db = np.zeros_like(self.Biases)

        # 计算填充
        if self.padding == "valid":
            pad = 0
        elif self.padding == "same":
            pad = (f - 1) // 2

        # 对输入进行填充
        A_prev_pad = np.pad(prev_layer_output, ((0, 0), (pad, pad),
                            (pad, pad), (0, 0)), mode='constant')

        for h in range(dA.shape[1]):  # 遍历输出高度
            for w in range(dA.shape[2]):  # 遍历输出宽度
                vert_start = h * self.stride[0]
                vert_end = vert_start + f
                horiz_start = w * self.stride[1]
                horiz_end = horiz_start + f

                # 提取当前窗口
                a_slice = A_prev_pad[:, vert_start:vert_end,
                                     horiz_start:horiz_end, :]

                # 对每个过滤器进行操作
                for c in range(n_C):
                    # 计算权重梯度
                    dW[:, :, :, c] += np.sum(a_slice * dA[:, h, w, c]
                                             [:, None, None, None], axis=0)

                    # 计算偏置梯度
                    db[c] += np.sum(dA[:, h, w, c])

                    # 计算输入梯度
                    dA_prev[:, vert_start:vert_end, horiz_start:horiz_end,
                            :] += self.Weights[:, :, :, c] * dA[:, h, w, c][:, None, None, None]

        # 裁剪掉填充
        if pad != 0:
            dA_prev = dA_prev[:, pad:-pad, pad:-pad, :]

        # 更新权重和偏置
        # self.Weights -= learning_rate * dW
        # self.Biases -= learning_rate * db

        self.Weights_Velocity = b1 * self.Weights_Velocity + (1 - b1) * dW
        self.Biases_Velocity = b1 * self.Biases_Velocity + (1 - b1) * db

        vdw_corrected = self.Weights_Velocity / (1 - b1 ** iter_num)
        vdb_corrected = self.Biases_Velocity / (1 - b1 ** iter_num)

        self.Squared_Weights = b2 * self.Squared_Weights + \
            (1 - b2) * np.square(vdw_corrected)
        self.Squared_Biases = b2 * self.Squared_Biases + \
            (1 - b2) * np.square(vdb_corrected)

        sdw_corrected = self.Squared_Weights / (1 - b2 ** iter_num)
        sdb_corrected = self.Squared_Biases / (1 - b2 ** iter_num)

        # perform gradient descent, update gradient
        self.Weights -= learning_rate * vdw_corrected / \
            np.sqrt(sdw_corrected + epsilon)
        self.Biases -= learning_rate * vdb_corrected / \
            np.sqrt(sdb_corrected + epsilon)
        return dA_prev

    def set_weights(self, weights):
        if weights.shape != self.Weights.shape:
            raise ValueError(
                f"Weights shape mismatch. Expected {self.Weights.shape}, got {weights.shape}")
        self.Weights = weights

    def set_bias(self, biases):
        if biases.shape != self.Biases.shape:
            raise ValueError(
                f"Biases shape mismatch. Expected {self.Biases.shape}, got {biases.shape}")
        self.Biases = biases

    def set_random_weights(self):
        self.Weights = np.random.randn(*self.Weights.shape)
        self.Biases = np.random.randn(*self.Biases.shape)

    def set_he_weights(self):
        # He初始化权重
        n = self.Weights.shape[1]  # 输入神经元数量
        scale = np.sqrt(2. / n)
        self.Weights = np.random.randn(*self.Weights.shape) * scale

        # 偏置通常初始化为0或很小的常数
        # self.Biases = np.zeros(self.Biases.shape)
        self.Biases = np.random.randn(*self.Biases.shape) / 1000
