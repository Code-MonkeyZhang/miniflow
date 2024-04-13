import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys


# append WorkSpace to sys.path
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

from src.util import *
from src.Layer_class import *
from src.Model_class import *

mnist = tf.keras.datasets.mnist
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

# Create Model
model = Model(
    [
        FlattenLayer(input_shape=(28, 28), name='Flatten'),
        Layer(25, activation="relu", name="L1"),
        Layer(10, activation='softmax', name="L2"),
    ], name="my_model")

# Load Weights
W1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_weights.npy', allow_pickle=True)
b1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_biases.npy', allow_pickle=True)
W2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_weights.npy', allow_pickle=True)
b2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_biases.npy', allow_pickle=True)

# Copy weights
# tensorflow输出的weights全是转置过的, 因为转置过的运算效率高
# 为了统一行和列分别代表 神经元数量和feature数量, 设置的时候需要进行转置
model.dense_array[1].set_weights(W1.T, b1)
model.dense_array[2].set_weights(W2.T, b2)

print(model.dense_array[0].get_weights().shape)
print(model.dense_array[1].get_weights().shape)
print(model.dense_array[2].get_weights().shape)

# Do prediction
predictions = model.predict(x_test)
print(predictions)
# Display Prediction
# 选择要可视化的样本数量
num_samples = 30
fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

for i, ax in enumerate(axes):
    # 选择一个随机索引
    idx = np.random.choice(x_test.shape[0])
    # 绘制图像
    ax.imshow(x_test[idx], cmap='gray')
    ax.axis('off')

    # 获取真实标签和预测标签
    true_label = y_test[idx]
    predicted_label = np.argmax(predictions[idx])

    # 设置标题显示真实标签和预测标签
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}")

plt.tight_layout()
plt.show()
