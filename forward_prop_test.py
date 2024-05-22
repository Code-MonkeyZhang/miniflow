import os
import numpy as np
from matplotlib import pyplot as plt

from src.miniflow import Model
from src.miniflow import Layer, FlattenLayer

# from miniflow import Layer, FlattenLayer

# Load data
# 设置数据文件的路径
data_dir = os.path.join(os.path.dirname(__file__), 'data/mnist_data')
x_train_path = os.path.join(data_dir, 'mnist_x_test.npy')
y_train_path = os.path.join(data_dir, 'mnist_y_train.npy')
x_test_path = os.path.join(data_dir, 'mnist_x_test.npy')
y_test_path = os.path.join(data_dir, 'mnist_y_test.npy')

# 加载测试集
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

############################## Create Model ########################################
Layer_list = [FlattenLayer(input_shape=(28, 28), layer_name='Flatten'),
              Layer(128, activation="relu", layer_name="L1", input_shape=784),
              Layer(64, activation="relu", layer_name="L2", input_shape=128),
              Layer(10, activation='softmax', layer_name="L3", input_shape=64), ]

############################## Load Weights ########################################
weights_path = "data/mnist_data/weights/"

# 遍历Layer_list中的每一层
for i, layer in enumerate(Layer_list):
    # 跳过FlattenLayer,因为它没有权重
    if isinstance(layer, FlattenLayer):
        continue

    # 构建权重和偏置文件的路径
    weights_file = weights_path + f"layer_{i - 1}_weights.npy"
    biases_file = weights_path + f"layer_{i - 1}_biases.npy"

    # 加载权重和偏置
    weights = np.load(weights_file)
    biases = np.load(biases_file)

    # 将加载的权重和偏置设置到当前层
    layer.set_weights(weights.T, biases)
    print(layer.get_weights().shape)


############################## Start Forward Prop ########################################

model = Model(
    [
        Layer_list[0],
        Layer_list[1],
        Layer_list[2],
        Layer_list[3]
    ], name="my_model", cost="softmax")

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

# 假设 predictions 是你的模型预测输出，现在你需要将其转换为类别标签
predictions = np.argmax(predictions, axis=1)  # 获取最大概率的索引作为预测标签

# 计算准确率
accuracy = np.mean(predictions == y_test)  # 比较预测标签和真实标签，计算准确率
print(f"Test Accuracy: {accuracy * 100:.2f}%")