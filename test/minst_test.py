import os
import numpy as np
from matplotlib import pyplot as plt
from miniflow import Model, Dense, FlattenLayer
# Load data

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设test文件夹在项目根目录下）
project_root = os.path.dirname(current_dir)
# 设置数据文件的路径
data_dir = os.path.join(project_root, 'data', 'mnist_data')
x_train_path = os.path.join(data_dir, 'mnist_x_train.npy')
y_train_path = os.path.join(data_dir, 'mnist_y_train.npy')
x_test_path = os.path.join(data_dir, 'mnist_x_test.npy')
y_test_path = os.path.join(data_dir, 'mnist_y_test.npy')

# 加载训练集
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)

# 加载测试集
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

############################## Create Model ########################################

model = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Dense(128, activation="relu", layer_name="L1", input_shape=784),
        Dense(64, activation="relu", layer_name="L2", input_shape=128),
        Dense(10, activation='softmax', layer_name="L3", input_shape=64),
    ], name="my_model", cost="softmax")

############################## Load the data ########################################

sample_size = 60000
x_samples = x_train[0:sample_size]
y_samples = y_train[0:sample_size]

############################## Train the model ########################################

model.summary()

model.dense_array[1].set_weights(np.load("weights/RandomWeights/L1_w.npy"), np.load("weights/RandomWeights/L1_b.npy"))
model.dense_array[2].set_weights(np.load("weights/RandomWeights/L2_w.npy"), np.load("weights/RandomWeights/L2_b.npy"))
model.dense_array[3].set_weights(np.load("weights/RandomWeights/L3_w.npy"), np.load("weights/RandomWeights/L3_b.npy"))

model.fit(x_samples, y_samples, learning_rate=0.002, epochs=10, batch_size=32, b1=0.9)
predictions = model.predict(x_test)

# 假设 predictions 是你的模型预测输出，现在你需要将其转换为类别标签
predictions = np.argmax(predictions, axis=1)  # 获取最大概率的索引作为预测标签

# 计算准确率
accuracy = np.mean(predictions == y_test)  # 比较预测标签和真实标签，计算准确率
print(f"Test Accuracy: {accuracy * 100:.2f}%")
