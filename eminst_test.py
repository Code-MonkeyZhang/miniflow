# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def load_data(filename):
    # 加载 CSV 文件
    data = pd.read_csv(filename)
    # 第一列是标签
    y = data.iloc[:, 0].values
    # 剩余列是图像数据
    X = data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # 归一化并调整形状
    return X, y

# 设置文件路径
train_file = '/home/yufeng/Workspace/miniflow/data/eminst/emnist-balanced-train.csv'
test_file = '/home/yufeng/Workspace/miniflow/data/eminst/emnist-balanced-test.csv'

# 加载数据
x_train, y_train = load_data(train_file)
x_test, y_test = load_data(test_file)

# 接下来，我们可以使用这些加载的数据来配置和训练你的 MiniFlow 模型
from src.miniflow import Model, Layer, FlattenLayer

model = Model(
    [
        FlattenLayer(input_shape=(28, 28, 1), layer_name='Flatten'),
        Layer(128, activation="relu", layer_name="L1"),
        Layer(64, activation="relu", layer_name="L2"),
        Layer(47, activation='softmax', layer_name="L3"),  # 注意：这里的输出层单元数应与标签数量一致
    ], name="emnist_model", cost="softmax")

model.set_rand_weight()

# 训练模型
model.fit(x_train, y_train, learning_rate=0.002, epochs=10, batch_size=32, b1=0.9)
predictions = model.predict(x_test)

# 计算准确率
predictions = np.argmax(predictions, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
