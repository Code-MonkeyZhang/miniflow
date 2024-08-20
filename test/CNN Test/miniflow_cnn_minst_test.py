import os
import numpy as np
from miniflow import Model, Dense, FlattenLayer, Conv2D, MaxPooling2D
from miniflow.util import label2onehot
import time

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

# Normalize and reshape
x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

# Convert y_train to OneHot
# 希望可以自己判断 one-hot 类别的数量
y_train = label2onehot(y_train, units=10)


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

# # load weights from file
# model.layers_array[0].set_weights(
#     np.load("./weights/simple_CNN_weights/conv2d_3x3_32_weights.npy"))
# model.layers_array[0].set_bias(
#     np.load("./weights/simple_CNN_weights/conv2d_3x3_32_biases.npy"))
#
# model.layers_array[2].set_weights(
#     np.load("./weights/simple_CNN_weights/conv2d_3x3_64_weights.npy"))
# model.layers_array[2].set_bias(
#     np.load("./weights/simple_CNN_weights/conv2d_3x3_64_biases.npy"))
#
# model.layers_array[5].set_weights(np.load("./weights/simple_CNN_weights/dense_64_weights.npy").T,
#                                   np.load("./weights/simple_CNN_weights/dense_64_biases.npy"))
# model.layers_array[6].set_weights(np.load("./weights/simple_CNN_weights/dense_10_weights.npy").T,
#                                   np.load("./weights/simple_CNN_weights/dense_10_biases.npy"))

model.set_rand_weight(method='He')


model.summary()

model.compile(optimizer='adam',
              alpha_decay=True,
              show_summary=True,
              plot_loss=True,
              loss_method="categorical_crossentropy")

train_example_num = 1000
model.fit(x_train[:train_example_num], y_train[:train_example_num],
          learning_rate=5e-5,
          epochs=10,
          batch_size=8,
          b1=0.9)

# Predictions using the trained model
test_example_num = 1000
start_time = time.time()
predictions = model.predict(x_test[:test_example_num])
end_time = time.time()

print(f"Prediction time: {end_time - start_time:.2f} seconds")

predictions = np.argmax(predictions, axis=1)
accuracy = np.mean(predictions == y_test[:test_example_num])
print(f"Test Accuracy: {accuracy * 100:.2f}%")
