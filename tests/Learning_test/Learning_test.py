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

############################## Create Model ########################################

model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Layer(25, activation="relu", layer_name="L1", input_shape=784),
        Layer(10, activation='softmax', layer_name="L2", input_shape=25),
    ], name="my_model", cost="softmax")

print(
    f"W1 shape = {model.dense_array[0].Weights.shape}, b1 shape = {model.dense_array[0].Biases.shape}")
print(
    f"W2 shape = {model.dense_array[1].Weights.shape}, b2 shape = {model.dense_array[1].Biases.shape}")
print(
    f"W3 shape = {model.dense_array[2].Weights.shape}, b3 shape = {model.dense_array[2].Biases.shape}")

############################## Load Weights ########################################
model.set_rand_weight()

# W1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_weights.npy', allow_pickle=True)
# b1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_biases.npy', allow_pickle=True)
# W2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_weights.npy', allow_pickle=True)
# b2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_biases.npy', allow_pickle=True)
#
# # Copy weights
# # tensorflow输出的weights全是转置过的, 因为转置过的运算效率高
# # 为了统一行和列分别代表 神经元数量和feature数量, 设置的时候需要进行转置
# model.dense_array[1].set_weights(W1.T, b1)
# model.dense_array[2].set_weights(W2.T, b2)

# 提取第一张图片及其标签
x_single = x_train[0:1]  # 保持批次维度，形状变为(1, 28, 28)
y_single = y_train[0:1]  # 保持批次维度，形状变为(1,)

sample_size = 60000
x_samples = x_train[0:sample_size]  # 提取前100个样本，形状变为(100, 28, 28)
y_samples = y_train[0:sample_size]  # 提取前100个标签，形状变为(100,)

############################## Train the model ########################################

model.fit(x_samples, y_samples, learningRate=0.00001, epochs=10)

# Do prediction
predictions = model.predict(x_test)

# Display Prediction
# 选择要可视化的样本数量
num_samples = 10
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
