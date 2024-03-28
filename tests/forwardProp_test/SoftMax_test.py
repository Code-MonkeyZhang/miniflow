import numpy as np
import tensorflow as tf
from src.util import *
from src.Layer_class import *
from src.Model_class import *
import tensorflow as tf
import matplotlib.pyplot as plt

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

print(
    f"W1 shape = {model.dense_array[0].Weights.shape}, b1 shape = {model.dense_array[0].Biases.shape}")
print(
    f"W2 shape = {model.dense_array[1].Weights.shape}, b2 shape = {model.dense_array[1].Biases.shape}")
print(
    f"W3 shape = {model.dense_array[2].Weights.shape}, b3 shape = {model.dense_array[2].Biases.shape}")

# Load Weights
W1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_weights.npy', allow_pickle=True)
b1 = np.load('../../data/minst_data/SoftMax_weights/layer_1_biases.npy', allow_pickle=True)
W2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_weights.npy', allow_pickle=True)
b2 = np.load('../../data/minst_data/SoftMax_weights/layer_2_biases.npy', allow_pickle=True)


# Copy weights
model.dense_array[1].set_weights(W1, b1)
model.dense_array[2].set_weights(W2, b2)

# Do prediction
predictions = model.predict(x_test)

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
