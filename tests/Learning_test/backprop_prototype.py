import numpy as np
import tensorflow as tf
import sys
import os


def find_project_root(current_dir, root_dir_name='Ynn'):
    """向上遍历目录树以找到项目根目录"""
    while True:
        parent_dir, dir_name = os.path.split(current_dir)
        if dir_name == root_dir_name:
            return current_dir
        if parent_dir == current_dir:  # 到达了文件系统的根目录
            raise FileNotFoundError(f"项目根目录 '{root_dir_name}' 未找到")
        current_dir = parent_dir


current_work_dir = os.getcwd()
project_root = find_project_root(current_work_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.Layer_class import *
from src.Model_class import *
from src.util import *

model = Model(
    [
        Layer(3, activation="relu", name="L1", input_shape=2),
        Layer(2, activation='softmax', name="L2", input_shape=3),
    ], name="my_model", cost="softmax")

"""
Setup weights and biases
"""

relu_weights = np.array([[1.0, 1.0],
                         [1.0, 3.0],
                         [1.0, 2.0]])

relu_biases = np.array([0.0, 0.0, 0.0])

softmax_weights = np.array([[0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2]])
softmax_biases = np.array([0.0, 0.0])

model.dense_array[0].set_weights(relu_weights, relu_biases)
model.dense_array[1].set_weights(softmax_weights, softmax_biases)

print(
    f"W1  = {model.dense_array[0].Weights},\n b1 shape = {model.dense_array[0].Biases}")
print(
    f"W2 = {model.dense_array[1].Weights}, \n b2 shape = {model.dense_array[1].Biases}")

"""
Start training 
"""

true_labels = np.array([[1.0, 0.0]])
train_x = np.array([[1.0, 2.0]])

model.fit(train_x, true_labels, 0.001, 1)
