import unittest
import numpy as np
from miniflow.util import *
from miniflow.Layer import Conv2D



class TestConv2D(unittest.TestCase):
    def setUp(self):
        # 创建输入图像 (1, 6, 6, 1)
        self.input_image = np.array([
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0]
        ]).reshape(1, 6, 6, 1)

        # 创建filter (3, 3, 1)
        self.Weights = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]).reshape(3, 3, 1)

        # 创建Conv2D层
        self.conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                                 input_shape=(6, 6, 1), stride=(1, 1), padding='valid')

        # 设置filter权重
        self.conv_layer.Weights = self.Weights.reshape(1, 3, 3, 1)
        self.conv_layer.Biases = np.array([0]).reshape(1, 1)

    def test_conv2d_output_shape(self):
        output = self.conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 4, 4, 1))

    def test_conv2d_output_values(self):
        output = self.conv_layer.compute_layer(self.input_image)
        expected_output = np.array([
            [0,  30,  30,  0],
            [0,  30,  30,  0],
            [0,  30,  30,  0],
            [0,  30,  30,  0]
        ])
        np.testing.assert_array_almost_equal(output.squeeze(), expected_output)

    def test_different_input_shapes(self):
        # 测试不同大小的输入
        input_image = np.random.rand(1, 8, 8, 1)
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(8, 8, 1), stride=(1, 1), padding='valid')
        output = conv_layer.compute_layer(input_image)
        self.assertEqual(output.shape, (1, 6, 6, 1))

    def test_multiple_filters(self):
        # 测试多个卷积核
        conv_layer = Conv2D(num_filter=3, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='valid')
        output = conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 4, 4, 3))

    def test_stride(self):
        # 测试不同的步长
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(2, 2), padding='valid')
        output = conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 2, 2, 1))

    def test_padding_same(self):
        # 测试 'same' 填充
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='same')
        output = conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 6, 6, 1))

    def test_multi_channel_input(self):
        # 测试多通道输入
        input_image = np.random.rand(1, 6, 6, 3)
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 3), stride=(1, 1), padding='valid')
        output = conv_layer.compute_layer(input_image)
        self.assertEqual(output.shape, (1, 4, 4, 1))

    def test_filter_initialization(self):
        # 测试卷积核的初始化
        conv_layer = Conv2D(num_filter=2, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='valid')
        self.assertEqual(conv_layer.Weights.shape, (2, 3, 3, 1))
        self.assertEqual(conv_layer.Biases.shape, (2, 1))

    def test_activation_function(self):
        def relu(x):
            return np.maximum(0, x)
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=relu, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='valid')
        conv_layer.Weights = self.Weights.reshape(1, 3, 3, 1)
        output = conv_layer.compute_layer(self.input_image)
        self.assertTrue(np.all(output >= 0))

    def test_bias_addition(self):
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='valid')
        conv_layer.Weights = self.Weights.reshape(1, 3, 3, 1)
        conv_layer.Biases = np.array([1]).reshape(1, 1)
        output = conv_layer.compute_layer(self.input_image)
        expected_output = np.array([
            [1,  31,  31,  1],
            [1,  31,  31,  1],
            [1,  31,  31,  1],
            [1,  31,  31,  1]
        ])
        np.testing.assert_array_almost_equal(output.squeeze(), expected_output)

    def test_multi_example_input(self):
        input_image = np.repeat(self.input_image, 3, axis=0)  # 3 examples
        output = self.conv_layer.compute_layer(input_image)
        self.assertEqual(output.shape, (3, 4, 4, 1))

    def test_large_stride(self):
        conv_layer = Conv2D(num_filter=1, kernel_size=(3, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(3, 3), padding='valid')
        conv_layer.Weights = self.Weights.reshape(1, 3, 3, 1)
        output = conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 2, 2, 1))

    def test_asymmetric_kernel(self):
        conv_layer = Conv2D(num_filter=1, kernel_size=(2, 3), activation=None, 
                            input_shape=(6, 6, 1), stride=(1, 1), padding='valid')
        conv_layer.Weights = np.random.randn(1, 2, 3, 1)
        output = conv_layer.compute_layer(self.input_image)
        self.assertEqual(output.shape, (1, 5, 4, 1))

    def test_conv_single_step(self):
        image_slice = np.array([
            [[1], [2], [3]],
            [[4], [5], [6]],
            [[7], [8], [9]]
        ])
        Weights = np.array([
            [[1], [0], [-1]],
            [[1], [0], [-1]],
            [[1], [0], [-1]]
        ])
        result = conv_single_step(image_slice, Weights)
        expected_result = (1*1 + 2*0 + 3*-1 + 4*1 + 5*0 + 6*-1 + 7*1 + 8*0 + 9*-1)
        self.assertAlmostEqual(result, expected_result)

            

if __name__ == '__main__':
    unittest.main(verbosity=2)