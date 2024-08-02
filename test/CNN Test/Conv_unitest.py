import unittest
import numpy as np
# Ensure to modify import statement as needed
from miniflow import Conv2D, MaxPooling2D


class TestConv2D(unittest.TestCase):
    def setUp(self):
        # Create a Conv2D instance with specific parameters
        self.conv2d = Conv2D(
            num_filter=1,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(6, 6, 1),
            stride=(1, 1),
            padding="valid",
        )

        # Set predefined weights (filter)
        filter_weights = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]]).reshape(
            3, 3, 1, 1
        )
        self.conv2d.set_weights(filter_weights)

        # Set bias to 0
        self.conv2d.set_bias(np.array([0]))

        # Create input image
        self.input_image = np.array(
            [
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
                [10, 10, 10, 0, 0, 0],
            ]
        ).reshape(1, 6, 6, 1)  # Add batch and channel dimensions

    def test_compute_layer(self):
        # Test: Verify the output of the convolution operation
        # It checks if the Conv2D layer correctly applies the filter to the input image
        result = self.conv2d.compute_layer(self.input_image)

        # Expected output after convolution
        expected_output = np.array(
            [
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0]
            ]
        ).reshape(1, 4, 4, 1)  # Add batch and channel dimensions

        # Verify the result matches the expected output
        np.testing.assert_array_equal(result, expected_output)


class TestMaxPooling2D(unittest.TestCase):
    def test_valid_padding_stride1(self):
        # Test: MaxPooling2D with valid padding and stride 1
        # It verifies the correct output for a 2x2 pooling over a 4x4 input with stride 1
        pool = MaxPooling2D(pool_size=(
            2, 2), input_shape=(4, 4, 1), stride=(1, 1))
        input_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
        ).reshape(1, 4, 4, 1)
        result = pool.compute_layer(input_data)
        expected_output = np.array([
            [6, 7, 8],
            [10, 11, 12],
            [14, 15, 16]]
        ).reshape(1, 3, 3, 1)
        np.testing.assert_array_equal(result, expected_output)

    def test_valid_padding_stride2(self):
        # Test: MaxPooling2D with valid padding and stride 2
        # It checks the correct output for a 2x2 pooling over a 4x4 input with stride 2
        pool = MaxPooling2D(pool_size=(
            2, 2), input_shape=(4, 4, 1), stride=(2, 2))
        input_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
        ).reshape(1, 4, 4, 1)
        result = pool.compute_layer(input_data)
        expected_output = np.array([[6, 8], [14, 16]]).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(result, expected_output)

    def test_larger_input(self):
        # Test: MaxPooling2D with a larger input
        # It verifies the correct output for a 3x3 pooling over a 5x5 input with stride 2
        pool = MaxPooling2D(pool_size=(
            3, 3), input_shape=(5, 5, 1), stride=(2, 2))
        input_data = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
        ).reshape(1, 5, 5, 1)
        result = pool.compute_layer(input_data)
        expected_output = np.array([[13, 15],
                                    [23, 25]]).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(result, expected_output)

    def test_output_shape(self):
        # Test: Verify the correct calculation of output shape for MaxPooling2D
        # It checks if the output_shape attribute is correctly set for different inputs and pool sizes
        pool = MaxPooling2D(pool_size=(
            2, 2), input_shape=(5, 5, 3), stride=(1, 1))
        self.assertEqual(pool.output_shape, (4, 4, 3))

        pool = MaxPooling2D(pool_size=(
            3, 3), input_shape=(7, 7, 2), stride=(2, 2))
        self.assertEqual(pool.output_shape, (3, 3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
