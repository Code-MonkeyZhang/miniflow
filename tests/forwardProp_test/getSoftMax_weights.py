import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

layers = model.layers
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()  # This gets both weights and biases, if they exist.
    if len(weights) > 0:  # Check if the layer has weights
        weight_file_path = f'layer_{i}_weights.npy'
        # np.save(weight_file_path, weights[0])  # Save the weights
        print(f"Saved weights to {weight_file_path}")

        if len(weights) > 1:  # Check if the layer has biases
            bias_file_path = f'layer_{i}_biases.npy'
            # np.save(bias_file_path, weights[1])  # Save the biases
            print(f"Saved biases to {bias_file_path}")

# 选择要可视化的样本数量
num_samples = 20
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
