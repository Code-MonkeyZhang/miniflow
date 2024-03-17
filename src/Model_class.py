from matplotlib import pyplot as plt
import numpy as np
import copy
import math
from typing import List, Callable
from src.util import *
from src.Layer_class import *

"""
======================================================================
Model Class
======================================================================
"""

'''
Use Case:
model = Sequential(
    [               
        Layer(25, activation="sigmoid", name="layer1"),
        Layer(15, activation="sigmoid", name="layer2"),
        Layer(1, activation="sigmoid", name="layer3"),
    ], name = "my_model" )
'''


class Model:
    def __init__(self, dense_array: List[Layer], name='model') -> None:
        self.dense_array = dense_array
        self.name = name

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
