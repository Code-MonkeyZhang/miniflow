import os
import numpy as np
from miniflow import Model, Dense, FlattenLayer, Conv2D, MaxPooling2D

x_train_path = './data/mnist_data/mnist_x_train.npy'
y_train_path = './data/mnist_data/mnist_y_train.npy'
x_test_path = './data/mnist_data/mnist_x_test.npy'
y_test_path = './data/mnist_data/mnist_y_test.npy'

# Load training set
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)

# Load test set
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

############################# Create Model ########################################
model = Model([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), input_shape=(26, 26, 32)),
    Conv2D(64, (3, 3), activation='relu', input_shape=(13, 13, 32)),
    MaxPooling2D((2, 2), input_shape=(11, 11, 64)),
    FlattenLayer(input_shape=(5, 5, 64)),
    Dense(64, activation='relu', input_shape=1600),
    Dense(10, activation='softmax', input_shape=64)
], name="my_model", cost="softmax")

# load weights from file
model.layers_array[0].set_weights(np.load("./weights/simple_CNN_weights/conv2d_3x3_32_weights.npy"))
model.layers_array[0].set_bias(np.load("./weights/simple_CNN_weights/conv2d_3x3_32_biases.npy"))

model.layers_array[2].set_weights(np.load("./weights/simple_CNN_weights/conv2d_3x3_64_weights.npy"))
model.layers_array[2].set_bias(np.load("./weights/simple_CNN_weights/conv2d_3x3_64_biases.npy"))

model.layers_array[5].set_weights(np.load("./weights/simple_CNN_weights/dense_64_weights.npy"),
                                  np.load("./weights/simple_CNN_weights/dense_64_biases.npy"))
model.layers_array[6].set_weights(np.load("./weights/simple_CNN_weights/dense_10_weights.npy"),
                                  np.load("./weights/simple_CNN_weights/dense_10_biases.npy"))

model.summary()
