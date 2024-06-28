import time
from typing import List
from .Layer_class import *
from .util import *

"""
======================================================================
Model Class
======================================================================
"""


class Model:
    def __init__(self, dense_array: List[Dense], cost, name='model') -> None:
        self.dense_array = dense_array
        self.layers_output = []
        self.name = name
        self.cost = cost
        self.iter_num = 0

        self.optimizer = None
        self.show_summary = None
        self.plot_loss = None
        self.alpha_decay = None

    # Iterate through each layer, and puts its output to the next layer
    def predict(self, x: np.ndarray) -> np.ndarray:
        prev_layer_output = x
        for dense_layer in self.dense_array:
            self.layers_output.append((dense_layer, prev_layer_output))
            layer_output = dense_layer.compute_layer(prev_layer_output)
            prev_layer_output = layer_output
        return prev_layer_output

    def compile(self, optimizer=None, alpha_decay=True, show_summary=False, plot_loss=False):
        self.optimizer = optimizer
        self.show_summary = show_summary
        self.plot_loss = plot_loss
        self.alpha_decay = alpha_decay

    def fit(self, X_train, y_train, learning_rate, epochs, batch_size=32,
            b1=0.2,
            b2=0.999,
            epsilon=1e-8,
            time_interval=1000,
            decay_rate=0.0002):

        epoch_lost_list = []
        epoch_time_list = []

        print("Start Training")
        for epoch in range(epochs):

            tic = time.time()
            epoch_lost = 0

            if self.alpha_decay:
                # Learning Rate Decay
                learning_rate = learning_rate / (1 + decay_rate * (epoch))

            print("Epoch {}/{}  ".format(epoch + 1, epochs))
            # Divide X_train into pieces, each piece is the size of batch size
            X_batch_list, y_batch_list = slice2batches(X_train, y_train, batch_size)

            batch_num = len(X_batch_list)

            # In each epoch, iterate through each batch
            for i in range(batch_num):

                print_progress_bar(i, batch_num)

                # Extract training example and label
                train_example = X_batch_list[i]
                label = y_batch_list[i]

                # Convert label to one-hot
                label_one_hot = np.zeros((label.shape[0], 10))
                label_one_hot[np.arange(
                    label.shape[0]), label] = 1  # use np advanced indexing to allocate the corresponding element to 1

                # Perform forward prop to compute the lost
                self.layers_output.clear()  # Clear the layers_output before Start
                prediction = self.predict(train_example)

                error = compute_cross_entropy_loss(prediction, label_one_hot)
                epoch_lost += error

                ###### START TRAINING #####

                # init backprop_gradient as all ones
                backprop_gradient = np.ones(self.dense_array[-1].get_weights().shape)

                self.iter_num += 1
                # reverse iterate layers Start backprop
                for layer, prev_layer_output in reversed(self.layers_output):
                    if layer.activation == "Flatten":
                        break  # ignore Flatten layer
                    backprop_gradient = layer.train_layer(prev_layer_output,
                                                          prediction,
                                                          label_one_hot,
                                                          learning_rate,
                                                          b1, b2, epsilon,
                                                          backprop_gradient,
                                                          self.iter_num)

            tok = time.time()
            epoch_time = 1000 * (tok - tic)
            epoch_time_list.append(epoch_time)

            epoch_lost = epoch_lost / batch_num
            epoch_lost_list.append(epoch_lost)

            print(" - Cost {:.6f} / Time {:.4f} ms".format(epoch_lost, epoch_time))

        if self.show_summary:
            train_summary(loss=epoch_lost_list, time=epoch_time_list)
        if self.plot_loss:
            plot_loss(epoch_lost_list)

    def set_rand_weight(self):
        for layer in self.dense_array:
            layer.set_random_weights()

    def save(self, path=""):
        for layer in self.dense_array:
            if layer.layer_name == "Flatten":
                continue
            np.save(path + layer.layer_name + "_w" + ".npy", layer.get_weights())
            np.save(path + layer.layer_name + "_b" + ".npy", layer.get_bias())

    def summary(self):
        """
        Print a summary of the model's layers, including name, type, weight shape, output shape, number of parameters, and activation.
        """
        total_params = 0
        print("Model Summary")
        print("=" * 120)
        print("{:30} {:20} {:20} {:20} {:10}".format("Layer (type)", "Weight Shape", "Output Shape", "Param #",
                                                     "Activation"))
        print("=" * 120)

        for layer in self.dense_array:
            # Get weight shape
            weight_shape = layer.Weights.shape if hasattr(layer, 'Weights') else 'No weights'

            # Assuming each layer has a `output_shape()` method that calculates its output shape
            output_shape = layer.output_shape() if hasattr(layer, 'output_shape') else 'Unknown'

            # Calculating parameters; this assumes layer has `count_params()` method
            params = layer.count_params() if hasattr(layer, 'count_params') else 0
            total_params += params

            # Prepare the layer activation, type, and name info
            layer_info = type(layer).__name__
            layer_name = layer.layer_name if hasattr(layer, 'layer_name') else 'Unnamed Layer'
            activation = getattr(layer, 'activation', 'None')

            # Print layer details
            print("{:30} {:20} {:20} {:20} {:10}".format(
                layer_name + ' (' + layer_info + ')',
                str(weight_shape),
                str(output_shape),
                str(params),
                activation
            ))

        print("=" * 120)
        print("Total params:", total_params)
        print("=" * 120)
