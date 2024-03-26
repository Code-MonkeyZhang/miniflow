import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.Model_class import *


model = Sequential(
    [
        ### START CODE HERE ###

        # layer1: The shape of W1 is (400, 25) and the shape of b1 is (25,)
        # layer2: The shape of W2 is (25, 15) and the shape of b2 is: (15,)
        # layer3: The shape of W3 is (15, 10) and the shape of b3 is: (10,)
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='relu', name="L1"),
        Dense(15, activation='relu', name="L2"),
        Dense(10, activation='linear', name="L3"),

        ### END CODE HERE ###
    ], name="my_model"
)

X = np.load("../../data/minst_data/X.npy")
y = np.load("../../data/minst_data/Y.npy")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X, y,
    epochs=40
)
[layer1, layer2, layer3] = model.layers
W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()

print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
np.save('../../data/minst_data/SoftMax_weights/W1.npy', W1)
np.save('../../data/minst_data/SoftMax_weights/b1.npy', b1)
np.save('../../data/minst_data/SoftMax_weights/W2.npy', W2)
np.save('../../data/minst_data/SoftMax_weights/b2.npy', b2)
np.save('../../data/minst_data/SoftMax_weights/W3.npy', W3)
np.save('../../data/minst_data/SoftMax_weights/b3.npy', b3)
print("Weights and biases are saved successfully.")