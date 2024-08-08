import tensorflow as tf
from tensorflow import keras
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 构建CNN模型
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
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

# # 获取每一层的权重和偏置，并保存
# for i, layer in enumerate(model.layers):
#     if layer.get_weights():  # 仅保存包含权重的层
#         weights = layer.get_weights()
#         layer_name = layer.__class__.__name__  # 获取层的类名
#         layer_config = layer.get_config()  # 获取层的配置
#
#         if isinstance(layer, keras.layers.Conv2D):
#             filters = layer_config['filters']
#             kernel_size = layer_config['kernel_size'][0]
#             filename = f'conv2d_{kernel_size}x{kernel_size}_{filters}'
#         elif isinstance(layer, keras.layers.Dense):
#             units = layer_config['units']
#             filename = f'dense_{units}'
#         else:
#             filename = f'layer_{i}'
#
#         np.save(f'./{filename}_weights.npy', weights[0])  # 权重
#         np.save(f'./{filename}_biases.npy', weights[1])   # 偏置
#
# print("权重和偏置已保存。")