import tensorflow as tf
from tensorflow import keras
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.reshape((60000, 28 * 28)) / 255.0
x_test = x_test.reshape((10000, 28 * 28)) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 构建模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# 获取每一层的权重和偏置
layer_weights = []
for layer in model.layers:
    weights = layer.get_weights()
    layer_weights.append(weights)

# 使用NumPy分别保存每一层的权重和偏置到不同的.npy文件
for i, weights in enumerate(layer_weights):
    np.save(f'./layer_{i}_weights.npy', weights[0])  # 权重
    np.save(f'./layer_{i}_biases.npy', weights[1])   # 偏置