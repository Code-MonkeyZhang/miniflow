import sys
from pathlib import Path
from src.util import *
import numpy as np

# Load Weights
W1 = np.load('data/minst_data/weights/W1.npy', allow_pickle=True)
W2 = np.load('data/minst_data/weights/W2.npy', allow_pickle=True)
W3 = np.load('data/minst_data/weights/W3.npy', allow_pickle=True)
b1 = np.load('data/minst_data/weights/b1.npy', allow_pickle=True)
b2 = np.load('data/minst_data/weights/b2.npy', allow_pickle=True)
b3 = np.load('data/minst_data/weights/b3.npy', allow_pickle=True)

print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

model = Model(
    [
        Dense(25, activation="sigmoid", name="layer1"),
        Dense(15, activation="sigmoid", name="layer2"),
        Dense(1, activation="sigmoid", name="layer3"),
    ], name="my_model")

model.dense_array[0].set_weights(W1, b1)
model.dense_array[1].set_weights(W2, b2)
model.dense_array[2].set_weights(W3, b3)

print(f"W1 shape = {model.dense_array[0].Weights.shape}, b1 shape = {model.dense_array[0].Biases.shape}")
print(f"W2 shape = {model.dense_array[1].Weights.shape}, b2 shape = {model.dense_array[1].Biases.shape}")
print(f"W3 shape = {model.dense_array[2].Weights.shape}, b3 shape = {model.dense_array[2].Biases.shape}")


# Load Data
def load_data():
    X = np.load(
        "/Users/zhangyufeng/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/WorkPlace/Personal Project/YNN/data/minst_data/X.npy")
    y = np.load(
        "/Users/zhangyufeng/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/WorkPlace/Personal Project/YNN/data/minst_data/y.npy")

    return X, y


# load dataset
X, y = load_data()
print(X.shape)
# Do prediction
Prediction = model.predict(X)
print(Prediction.shape)

Yhat = (Prediction >= 0.5).astype(int)
print("predict a zero: ", Yhat[0], "predict a one: ", Yhat[500])

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]}, {Yhat[random_index, 0]}")
    ax.set_axis_off()
fig.suptitle("Label, Yhat", fontsize=16)
plt.show()
