import numpy as np
import pdb;
from src.Layer_class import *

"""
======================================================================
Model Class
======================================================================
"""


class Model:
    def __init__(self, dense_array: List[Layer], cost, name='model') -> None:
        self.dense_array = dense_array
        self.layers_output = []
        self.name = name
        self.cost = cost

    # Iterate through each layer, and puts its output to the next layer
    def predict(self, x: np.ndarray) -> np.ndarray:
        prev_layer_output = x
        for dense_layer in self.dense_array:
            self.layers_output.append((dense_layer, prev_layer_output))
            layer_output = dense_layer.compute_layer(prev_layer_output)
            prev_layer_output = layer_output
            if np.any(np.isnan(prev_layer_output)):
                print(f"NaN detected in layer_output")
                a = 1 + 1
                print(a)
        return prev_layer_output

    def fit(self, X_train, y_train, learningRate, epochs):
        # perform backward prop
        print("Start Training")
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))

            # In each epoch we iterate through each training example
            for i in range(X_train.shape[0]):

                # Make sure to clear the layers_output at the start of processing each sample
                self.layers_output.clear()

                train_example = X_train[i][np.newaxis, ...]
                label = y_train[i]

                # do forward prop to compute lost
                # and save output of each layer
                prediction = self.predict(train_example)

                loss = self.compute_loss(prediction, label)
                print("Loss: {}".format(loss))

                # start training
                # init backprop_gradient as all ones, when training from the last layer(output layer)
                backprop_gradient = np.ones(self.dense_array[-1].get_weights().shape)

                # reverse iterate the layers
                for layer, prev_layer_output in reversed(self.layers_output):
                    if layer.activation == "Flatten":
                        break
                    backprop_gradient = layer.train_layer(prev_layer_output, prediction, label, learningRate,
                                                          backprop_gradient)

    def compute_loss(self, prediction, target):
        # Compute the loss between the predicted output and the target
        # You can use any loss function here, such as mean squared error
        loss = np.mean((prediction - target))
        return loss

    def set_rand_weight(self):
        for layer in self.dense_array:
            layer.set_random_weights()
        print("Set random weights Complete")

# def evaluate(self):
#     pass
# def get_weights(self):
#     pass
# use forward prop to get prediction
