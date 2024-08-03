import unittest
import numpy as np
# Ensure to modify import statement as needed
from miniflow import Conv2D, MaxPooling2D

import unittest
from unittest.runner import TextTestResult
from unittest.signals import registerResult
import time

# Define edge detection filters
vertical_edge_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]).reshape(3, 3, 1, 1)

horizontal_edge_filter = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]).reshape(3, 3, 1, 1)

diagonal_edge_filter_1 = np.array([
    [-1, 0, 1],
    [0, 0, 0],
    [1, 0, -1]
]).reshape(3, 3, 1, 1)

diagonal_edge_filter_2 = np.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
]).reshape(3, 3, 1, 1)

spot_detection_filter = np.array([
    [1, -1, 1],
    [-1, 1, -1],
    [1, -1, 1]
]).reshape(3, 3, 1, 1)


class CustomTestResult(TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions

    def startTest(self, test):
        super().startTest(test)
        if self.showAll:
            self.stream.write(
                f"{test.__class__.__name__}.{test._testMethodName}: ")
            self.stream.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.writeln("OK")
        elif self.dots:
            self.stream.write('.')
            self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln("ERROR")
        elif self.dots:
            self.stream.write('E')
            self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln("FAIL")
        elif self.dots:
            self.stream.write('F')
            self.stream.flush()


class CustomTestRunner(unittest.TextTestRunner):
    resultclass = CustomTestResult

    def run(self, test):
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals
        startTime = time.time()
        startTestRun = getattr(result, 'startTestRun', None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, 'stopTestRun', None)
            if stopTestRun is not None:
                stopTestRun()
        stopTime = time.time()
        timeTaken = stopTime - startTime
        result.printErrors()
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(len, (result.expectedFailures,
                                result.unexpectedSuccesses,
                                result.skipped))
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        return result


# Create input image
# Set weights & bias
# Verify the result


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

    def simple_vertical_edge_test(self):
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

        # Set weights & bias
        vertical_edge_filter = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]).reshape(3, 3, 1, 1)
        self.conv2d.set_weights(vertical_edge_filter)
        self.conv2d.set_bias(np.array([0]))

        # Verify the result
        result = self.conv2d.compute_layer(self.input_image)
        expected_output = np.array(
            [
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0]
            ]
        ).reshape(1, 4, 4, 1)  # Add batch and channel dimensions

        np.testing.assert_array_equal(result, expected_output)

    def simple_horizontal_edge_test(self):

        # Create a horizontal edge input image
        horizontal_edge_image = np.array([
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]).reshape(1, 6, 6, 1)  # Add batch and channel dimensions

        # Set weights & bias
        horizontal_edge_filter = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ]).reshape(3, 3, 1, 1)
        self.conv2d.set_weights(horizontal_edge_filter)

        # Verify the result
        result = self.conv2d.compute_layer(horizontal_edge_image)
        expected_output = np.array([
            [0, 0, 0, 0],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [0, 0, 0, 0]
        ]).reshape(1, 4, 4, 1)

        np.testing.assert_array_equal(result, expected_output)

    # Test multiple filters
    def multiple_filters_test(self):
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

        # Set weights & bias
        filters = np.array([
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]],

            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],

            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]],
        ]).reshape(3, 3, 1, 3)

        self.conv2d.set_weights(filters)
        self.conv2d.set_bias(np.array([0, 0, 0]))  # 修改这里

        # Verify the result
        result = self.conv2d.compute_layer(self.input_image)
        expected_output = np.array(
            [
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0],
                [0, 30, 30, 0]
            ]
        ).reshape(1, 4, 4, 1)  # Add batch and channel dimensions

        # Calculate expected output
        expected_output = np.zeros((1, 4, 4, 3))

        # Filter 1: Vertical edge detection
        expected_output[0, :, :, 0] = np.array([
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0]
        ])

        # Filter 2: Horizontal edge detection
        expected_output[0, :, :, 1] = np.array([
            [0, 0, 0, 0],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [0, 0, 0, 0]
        ])

        # Filter 3: Spot detection
        expected_output[0, :, :, 2] = np.array([
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0],
            [0, 30, 30, 0]
        ])

        # Test if the computed result matches the expected output
        np.testing.assert_array_almost_equal(
            result, expected_output, decimal=6)

    # Test multiple channels

    def multiple_channels_test(self):
        # Create a Conv2D instance with multiple input channels
        conv2d_multi = Conv2D(
            num_filter=2,
            kernel_size=(2, 2),
            activation="relu",
            input_shape=(3, 3, 2),  # 2 input channels
            stride=(1, 1),
            padding="valid",
        )

        # Create a simple input image with 2 channels
        input_image = np.array([
            [[[1, 1], [2, 2], [3, 3]],
             [[4, 4], [5, 5], [6, 6]],
             [[7, 7], [8, 8], [9, 9]]]
        ])

        # Set weights & bias
        weights = np.array([
            [
                [[1, 0], [0, 1]],
                [[1, 0], [0, 1]]
            ],
            [
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]]
            ]
        ])

        conv2d_multi.set_weights(weights)
        conv2d_multi.set_bias(np.array([0, 0]))

        # Compute the result
        result = conv2d_multi.compute_layer(input_image)

        # Calculate expected output
        expected_output = np.zeros((1, 2, 2, 2))

        # Filter 1: Diagonal elements filter
        expected_output[0, :, :, 0] = np.array([
            [12, 16],
            [28, 32]
        ])

        # Filter 2: Sum filter (only first channel)
        expected_output[0, :, :, 1] = np.array([
            [12, 16],
            [24, 28]
        ])

        # Test if the computed result matches the expected output
        np.testing.assert_array_almost_equal(
            result, expected_output, decimal=6)


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

    def test_output_shape_stride1(self):
        # Test: Verify the correct calculation of output shape for MaxPooling2D with stride 1
        pool = MaxPooling2D(pool_size=(
            2, 2), input_shape=(10, 10, 3), stride=(1, 1))
        self.assertEqual(pool.output_shape, (9, 9, 3))

    def test_output_shape_stride2(self):
        # Test: Verify the correct calculation of output shape for MaxPooling2D with stride 2
        pool = MaxPooling2D(pool_size=(
            2, 2), input_shape=(8, 8, 4), stride=(2, 2))
        self.assertEqual(pool.output_shape, (4, 4, 4))

    def test_output_shape_rectangular_input(self):
        # Test: Verify the correct calculation of output shape for MaxPooling2D with rectangular input
        pool = MaxPooling2D(pool_size=(
            3, 3), input_shape=(15, 25, 2), stride=(2, 2))
        self.assertEqual(pool.output_shape, (7, 12, 2))

    def test_output_shape_uneven_pool_size(self):
        # Test: Verify the correct calculation of output shape for MaxPooling2D with uneven pool size
        pool = MaxPooling2D(pool_size=(
            2, 3), input_shape=(12, 12, 1), stride=(1, 1))
        self.assertEqual(pool.output_shape, (11, 10, 1))


if __name__ == "__main__":
    unittest.main(testRunner=CustomTestRunner(verbosity=2))
