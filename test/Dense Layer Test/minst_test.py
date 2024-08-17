import os
import numpy as np
import miniflow
from matplotlib import pyplot as plt
from miniflow import Model, Dense, FlattenLayer

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
        Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L2", input_shape=128),
        Dense(10, activation='softmax', layer_name="L3", input_shape=64),
    ], name="my_model", cost="softmax")

############################## Load the data ########################################

sample_size = 60000
x_samples = x_train[0:sample_size]
y_samples = y_train[0:sample_size]

############################## Train the model ########################################

# Set initial weights from stored files
model.layers_array[1].set_weights(np.load("./weights/RandomWeights/L1_w.npy"),
                                  np.load("./weights/RandomWeights/L1_b.npy"))
model.layers_array[2].set_weights(np.load("./weights/RandomWeights/L2_w.npy"),
                                  np.load("./weights/RandomWeights/L2_b.npy"))
model.layers_array[3].set_weights(np.load("./weights/RandomWeights/L3_w.npy"),
                                  np.load("./weights/RandomWeights/L3_b.npy"))

# Compile the model with settings
model.compile(optimizer='adam',
              alpha_decay=True,
              show_summary=False,
              plot_loss=False,
              loss_method="categorical_crossentropy")

model.summary()

model.fit(x_samples,
          y_samples,
          learning_rate=0.002,
          epochs=10,
          batch_size=32,
          b1=0.9)

# Predictions using the trained model
predictions = model.predict(x_test)

# Assume 'predictions' are the output of your model, now you need to convert it to class labels
# Get the index of the highest probability as the prediction label
predictions = np.argmax(predictions, axis=1)

# Calculate accuracy
# Compare predicted labels with actual labels to calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
