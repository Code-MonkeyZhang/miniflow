# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from miniflow import Model, Dense, FlattenLayer


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

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

sample_size = 60000
x_samples = x_train[0:sample_size]
y_samples = y_train[0:sample_size]

model = Model(
    [
        FlattenLayer(input_shape=(28, 28, 1), layer_name='Flatten'),
        Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L2", input_shape=128),
        Dense(47, activation='softmax', layer_name="L3", input_shape=64),  # 注意：这里的输出层单元数应与标签数量一致
    ], name="emnist_model", cost="softmax")

# model.set_rand_weight()

model.dense_array[1].set_weights(
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Dense_1_weights.npy"),
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Dense_1_biases.npy"))
model.dense_array[2].set_weights(
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Dense_2_weights.npy"),
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Dense_2_biases.npy"))
model.dense_array[3].set_weights(
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Softmax_weights.npy"),
    np.load("/home/yufeng/Workspace/miniflow/weights/tf_eminst_rand_weights/layer_Softmax_biases.npy"))

model.summary()

model.compile(optimizer='adam',
              alpha_decay=True,
              show_summary=True,
              plot_loss=True,
              )

# 训练模型
model.fit(x_train, y_train, learning_rate=0.0003, epochs=10, batch_size=32, b1=0.9)
predictions = model.predict(x_test)

# 计算准确率
predictions = np.argmax(predictions, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
