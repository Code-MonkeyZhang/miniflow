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

mnist = tf.keras.datasets.mnist
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

# Print Data
print(x_train.shape)
print(y_train.shape)

# Create Model

model = Model(
    [
        FlattenLayer(input_shape=(28, 28), name='Flatten'),
        Layer(25, activation="relu", name="L1", input_shape=784),
        Layer(10, activation='softmax', name="L2", input_shape=25),
    ], name="my_model",cost="softmax")

print(
    f"W1 shape = {model.dense_array[0].Weights.shape}, b1 shape = {model.dense_array[0].Biases.shape}")
print(
    f"W2 shape = {model.dense_array[1].Weights.shape}, b2 shape = {model.dense_array[1].Biases.shape}")
print(
    f"W3 shape = {model.dense_array[2].Weights.shape}, b3 shape = {model.dense_array[2].Biases.shape}")

# 提取第一张图片及其标签
x_single = x_train[0:1]  # 保持批次维度，形状变为(1, 28, 28)
y_single = y_train[0:1]  # 保持批次维度，形状变为(1,)


model.set_rand_weight()

# Train the model
model.fit(x_train, y_train, learningRate=0.0001, epochs=1)
