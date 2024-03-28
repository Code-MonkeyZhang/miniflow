from src.Layer_class import *

"""
======================================================================
Model Class
======================================================================
"""

class Model:
    def __init__(self, dense_array: List[Layer], name='model') -> None:
        self.dense_array = dense_array
        self.name = name

    # Iterate through each layer, and puts its output to the next layer
    def predict(self, x: np.ndarray) -> np.ndarray:
        for dense_layer in self.dense_array:
            x = dense_layer.compute_layer(x)
        return x

    # TBD
    # def fit(self, X_train, y_train, epochs):
    #     # perform backward prop
    #     pass
    # def evaluate(self):
    #     pass
    # def get_weights(self):
    #     pass
