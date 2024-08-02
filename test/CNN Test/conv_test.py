import unittest
import numpy as np
from miniflow import Conv2D,MaxPooling2D  # 请确保根据实际情况修改导入语句


class TestConv2D(unittest.TestCase):
    def setUp(self):
        # 创建一个Conv2D实例
        self.conv2d = Conv2D(num_filter=1, kernel_size=(3, 3), activation='relu',
                             input_shape=(6, 6, 1), stride=(1, 1), padding='valid')

        # 设置权重（滤波器）
        filter_weights = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]).reshape(3, 3, 1, 1)
        self.conv2d.set_weights(filter_weights)

        # 设置偏置为0
        self.conv2d.set_bias(np.array([0]))

        # 创建输入图像
        self.input_image = np.array([
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0]
        ]).reshape(1, 6, 6, 1)  # 添加批次和通道维度

    def test_compute_layer(self):
        # 计算卷积结果
        result = self.conv2d.compute_layer(self.input_image)

        # 期望的输出
        expected_output = np.array([
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0]
        ]).reshape(1, 4, 4, 1)  # 添加批次和通道维度

        # 验证结果
        np.testing.assert_array_equal(result, expected_output)




class TestMaxPooling2D(unittest.TestCase):
    def test_valid_padding_stride1(self):
        # 创建一个 MaxPooling2D 实例
        pool = MaxPooling2D(pool_size=(2, 2), input_shape=(4, 4, 1), stride=(1, 1), padding='valid')

        # 创建输入
        input_data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]).reshape(1, 4, 4, 1)

        # 计算池化结果
        result = pool.compute_layer(input_data)

        # 期望的输出
        expected_output = np.array([
            [6, 7, 8],
            [10, 11, 12],
            [14, 15, 16]
        ]).reshape(1, 3, 3, 1)

        # 验证结果
        np.testing.assert_array_equal(result, expected_output)


if __name__ == '__main__':
    unittest.main()
