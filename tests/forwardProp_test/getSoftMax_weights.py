import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(25, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

layers = model.layers
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()  # This gets both weights and biases, if they exist.
    if len(weights) > 0:  # Check if the layer has weights
        weight_file_path = f'layer_{i}_weights.npy'
        # np.save(weight_file_path, weights[0])  # Save the weights
        print(f"Saved weights to {weight_file_path}")

        if len(weights) > 1:  # Check if the layer has biases
            bias_file_path = f'layer_{i}_biases.npy'
            # np.save(bias_file_path, weights[1])  # Save the biases
            print(f"Saved biases to {bias_file_path}")
