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
    MaxPooling2D((2, 2)), # type: ignore
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    FlattenLayer(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# load weights from file
model.

model.summary()



