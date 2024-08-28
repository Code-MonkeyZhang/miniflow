import time
from typing import List
from .Layer.Dense_Layer import Dense
from .Layer.Flatten_Layer import FlattenLayer
from .Layer.MaxPooling2D import MaxPooling2D
from .Layer.Conv2D_Layer import Conv2D
from .util import *

"""
======================================================================
Model Class
======================================================================
"""


class Model:

    def __init__(self, layers_array: List[Dense], cost, name='model') -> None:
        self.layers_array = layers_array
        self.layers_output = []
        self.name = name
        self.cost = cost
        self.iter_num = 0

        self.optimizer = None
        self.show_summary = None
        self.plot_loss = None
        self.alpha_decay = None
        self.loss_method = None

    # Iterate through each layer, and puts its output to the next layer
    def predict(self, x: np.ndarray) -> np.ndarray:
        prev_layer_output = x
        for layer in self.layers_array:
            self.layers_output.append((layer, prev_layer_output))
            layer_output = layer.compute_layer(prev_layer_output)
            prev_layer_output = layer_output
        return prev_layer_output

    def compile(self, optimizer=None, alpha_decay=True, show_summary=False, plot_loss=False, loss_method=""):
        self.optimizer = optimizer
        self.show_summary = show_summary
        self.plot_loss = plot_loss
        self.alpha_decay = alpha_decay
        self.loss_method = loss_method

    def fit(self, X_train, y_train, learning_rate, epochs,
            batch_size=32,
            b1=0.2,
            b2=0.999,
            epsilon=1e-8,
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
            X_batch_list, y_batch_list = slice2batches(
                X_train, y_train, batch_size)

            batch_num = len(X_batch_list)

            # In each epoch, iterate through each batch
            for i in range(batch_num):

                print_progress_bar(i, batch_num)

                # Extract training example and label
                train_example = X_batch_list[i]
                label = y_batch_list[i]

                ############################## Forward PROP ########################################

                self.layers_output.clear()  # Clear the layers_output before Start
                prediction = self.predict(train_example)
                error = self.compute_error(
                    prediction=prediction, label=label, loss_method=self.loss_method)
                epoch_lost += error

                ############################## START TRAINING ########################################

                # init backprop_gradient as all ones
                backprop_gradient = np.ones(
                    self.layers_array[-1].get_weights().shape)

                self.iter_num += 1
                # reverse iterate layers Start backprop
                for layer, prev_layer_output in reversed(self.layers_output):
                    layer_name = layer.__class__.__name__
                    if isinstance(layer, FlattenLayer) or isinstance(layer, MaxPooling2D):
                        # 如果是 FlattenLayer，执行它的反向传播方法
                        backprop_gradient = layer.backward_prop(
                            dA=backprop_gradient)
                    elif isinstance(layer, Conv2D):
                        backprop_gradient = layer.backward_prop(prev_layer_output,
                                                                backprop_gradient,
                                                                learning_rate,
                                                                b1, b2, epsilon,
                                                                self.iter_num)

                    elif isinstance(layer, Dense):
                        backprop_gradient = layer.backward_prop(prev_layer_output,
                                                                prediction,
                                                                label,
                                                                learning_rate,
                                                                b1, b2, epsilon,
                                                                backprop_gradient,
                                                                self.iter_num)

            tok = time.time()
            epoch_time = 1000 * (tok - tic)
            epoch_time_list.append(epoch_time)

            epoch_lost = epoch_lost / batch_num
            epoch_lost_list.append(epoch_lost)

            print(
                " - Cost {:.6f} / Time {:.4f} ms".format(epoch_lost, epoch_time))

        if self.show_summary:
            train_summary(loss=epoch_lost_list, time=epoch_time_list)
        if self.plot_loss:
            plot_loss(epoch_lost_list)

    def compute_error(self, prediction, label, loss_method):
        if loss_method == "categorical_crossentropy":
            return compute_cross_entropy_loss(prediction, label)

    def set_rand_weight(self, method='He'):
        for layer in self.layers_array:
            # Skip FlattenLayer and MaxPooling2D Because they don't have weights
            if type(layer).__name__ == "FlattenLayer":
                continue
            if type(layer).__name__ == "MaxPooling2D":
                continue

            if method == 'He':
                layer.set_he_weights()
            elif method == 'Xavier':
                # layer.set_xavier_weights()
                pass
            elif method == 'Random':
                layer.set_random_weights()

    def save_params(self, path=""):
        for layer in self.layers_array:
            # Skip FlattenLayer and MaxPooling2D Because they don't have weights
            if type(layer).__name__ == "FlattenLayer":
                continue
            if type(layer).__name__ == "MaxPooling2D":
                continue
            np.save(path + layer.layer_name + "_w" +
                    ".npy", layer.get_weights())
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

        for layer in self.layers_array:
            # Get weight shape
            weight_shape = layer.Weights.shape if hasattr(
                layer, 'Weights') else 'No weights'

            # Assuming each layer has a `output_shape()` method that calculates its output shape
            output_shape = layer.get_output_shape() if hasattr(
                layer, 'output_shape') else 'Unknown'

            # Calculating parameters; this assumes layer has `count_params()` method
            params = layer.count_params() if hasattr(layer, 'count_params') else 0
            total_params += params

            # Prepare the layer activation, type, and name info
            layer_info = type(layer).__name__
            layer_name = layer.layer_name if hasattr(
                layer, 'layer_name') else 'Unnamed Layer'
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
