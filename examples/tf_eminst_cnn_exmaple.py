import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def load_data(filename):
    data = pd.read_csv(filename)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # 归一化并调整形状
    y = tf.keras.utils.to_categorical(y, num_classes=47)  # 转换为 one-hot 编码
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

# 定义基于AlexNet的模型
model = Sequential([
    Conv2D(96, (3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    
    Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    
    Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    
    Dense(4096, activation='relu'),
    Dropout(0.5),
    
    Dense(47, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0001),  # 降低学习率
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")