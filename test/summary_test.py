import os
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/home/yufeng/Workspace/miniflow')
from miniflow import Model, Dense, FlattenLayer


############################## Create Model ########################################

model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L2", input_shape=128),
        Dense(10, activation='softmax', layer_name="L3", input_shape=64),
    ], name="my_model", cost="softmax")

############################## Load the data ########################################
model.dense_array[1].set_weights(np.load("weights/RandomWeights/L1_w.npy"), np.load("weights/RandomWeights/L1_b.npy"))
model.dense_array[2].set_weights(np.load("weights/RandomWeights/L2_w.npy"), np.load("weights/RandomWeights/L2_b.npy"))
model.dense_array[3].set_weights(np.load("weights/RandomWeights/L3_w.npy"), np.load("weights/RandomWeights/L3_b.npy"))

############################## Train the model ########################################

model.summary()
