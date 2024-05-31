import os
import numpy as np

from src.miniflow import Model
from src.miniflow import Layer, FlattenLayer

# from miniflow import Layer, FlattenLayer

# Load data
# 设置数据文件的路径
data_dir = os.path.join(os.path.dirname(__file__), 'data/mnist_data')
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
        Layer(64, activation="relu", layer_name="L1", input_shape=784),
        Layer(10, activation='softmax', layer_name="L2", input_shape=64),
    ], name="my_model", cost="softmax")

model2 = Model(
    [
        FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
        Layer(128, activation="relu", layer_name="L1", input_shape=784),
        Layer(64, activation="relu", layer_name="L2", input_shape=128),
        Layer(10, activation='softmax', layer_name="L3", input_shape=64),
    ], name="my_model", cost="softmax")

############################## Load the data ########################################

sample_size = 60000
x_samples = x_train[0:sample_size]
y_samples = y_train[0:sample_size]

############################## Train the model ########################################
# model.set_rand_weight()
# model.fit(x_samples, y_samples, learningRate=0.05, epochs=5)
# predictions = model.predict(x_test)

model2.set_rand_weight()
model2.fit(x_samples, y_samples, learning_rate=0.002, epochs=5, batch_size=32)
predictions = model2.predict(x_test)

# # Display Prediction
# # 选择要可视化的样本数量
# num_samples = 30
# fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
#
# for i, ax in enumerate(axes):
#     # 选择一个随机索引
#     idx = np.random.choice(x_test.shape[0])
#     # 绘制图像
#     ax.imshow(x_test[idx], cmap='gray')
#     ax.axis('off')
#
#     # 获取真实标签和预测标签
#     true_label = y_test[idx]
#     predicted_label = np.argmax(predictions[idx])
#
#     # 设置标题显示真实标签和预测标签
#     ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
#
# plt.tight_layout()
# plt.show()

# 假设 predictions 是你的模型预测输出，现在你需要将其转换为类别标签
predictions = np.argmax(predictions, axis=1)  # 获取最大概率的索引作为预测标签

# 计算准确率
accuracy = np.mean(predictions == y_test)  # 比较预测标签和真实标签，计算准确率
print(f"Test Accuracy: {accuracy * 100:.2f}%")
