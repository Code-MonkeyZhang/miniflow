import os
import numpy as np
import miniflow
from matplotlib import pyplot as plt
from miniflow import Model, Dense, FlattenLayer
import os

# 设置当前工作目录为项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
print(f"Current Working Directory: {os.getcwd()}")


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
y_train = miniflow.label2onehot(y_train, units=10)

############################## Create Model ########################################

model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        # Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L1", input_shape=784),
        Dense(10, activation='softmax', layer_name="L2", input_shape=64),
    ], name="my_model", cost="softmax")

############################## Load the data ########################################

train_example_num = 1000
x_samples = x_train[0:train_example_num]
y_samples = y_train[0:train_example_num]

############################## Train the model ########################################

model.set_rand_weight(method='He')

# Compile the model with settings
model.compile(optimizer='adam',
              alpha_decay=True,
              show_summary=False,
              plot_loss=False,
              loss_method="categorical_crossentropy")

model.summary()

model.fit(x_samples,
          y_samples,
          learning_rate=5e-5,
          epochs=10,
          batch_size=8,
          b1=0.9)

test_example_num = 1000


# Predictions using the trained model
predictions = model.predict(x_test[:test_example_num])

predictions = np.argmax(predictions, axis=1)

accuracy = np.mean(predictions == y_test[:test_example_num])
print(f"Test Accuracy: {accuracy * 100:.2f}%")
