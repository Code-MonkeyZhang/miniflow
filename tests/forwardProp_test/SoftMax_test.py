import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from src.util import *
from src.Layer_class import *
from src.Model_class import *

X = np.load("../../data/minst_data/X.npy")
y = np.load("../../data/minst_data/Y.npy")

# Load Weights
W1 = np.load('../../data/minst_data/SoftMax_weights/W1.npy', allow_pickle=True)
W2 = np.load('../../data/minst_data/SoftMax_weights/W2.npy', allow_pickle=True)
W3 = np.load('../../data/minst_data/SoftMax_weights/W3.npy', allow_pickle=True)
b1 = np.load('../../data/minst_data/SoftMax_weights/b1.npy', allow_pickle=True)
b2 = np.load('../../data/minst_data/SoftMax_weights/b2.npy', allow_pickle=True)
b3 = np.load('../../data/minst_data/SoftMax_weights/b3.npy', allow_pickle=True)

model = Model(
    [
        Layer(25, activation="sigmoid", name="layer1"),
        Layer(15, activation="sigmoid", name="layer2"),
        Layer(1, activation="sigmoid", name="layer3"),
    ], name="my_model")

model.dense_array[0].set_weights(W1, b1)
model.dense_array[1].set_weights(W2, b2)
model.dense_array[2].set_weights(W3, b3)

print(
    f"W1 shape = {model.dense_array[0].Weights.shape}, b1 shape = {model.dense_array[0].Biases.shape}")
print(
    f"W2 shape = {model.dense_array[1].Weights.shape}, b2 shape = {model.dense_array[1].Biases.shape}")
print(
    f"W3 shape = {model.dense_array[2].Weights.shape}, b3 shape = {model.dense_array[2].Biases.shape}")
