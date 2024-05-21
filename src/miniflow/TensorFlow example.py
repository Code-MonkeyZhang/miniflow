import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 预处理数据
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化到0到1之间

# 展平图像
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 构建全连接神经网络模型
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# 可视化训练过程
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 进行预测
predictions = model.predict(x_test)

# 可视化部分预测结果
num_samples = 30
fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
for i, ax in enumerate(axes):
    idx = np.random.choice(x_test.shape[0])
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.axis('off')
    true_label = y_test[idx]
    predicted_label = np.argmax(predictions[idx])
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
plt.tight_layout()
plt.show()
