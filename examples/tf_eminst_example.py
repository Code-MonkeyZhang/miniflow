import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time

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

# 定义模型
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # 平坦层
    Dense(256, activation='relu'),  # 增加到256个神经元
    Dropout(0.5),  # 添加Dropout来减少过拟合
    Dense(128, activation='relu'),  # 第二个全连接层
    Dropout(0.3),  # 进一步添加Dropout
    Dense(47, activation='softmax')  # 输出层，47个类别
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),  # 提高学习率
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 自定义回调函数来记录每个 epoch 的时间
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        print(f"Epoch {epoch + 1} took {self.times[-1]:.2f} seconds")

# 创建回调实例
time_callback = TimeHistory()

# 训练模型
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), callbacks=[time_callback])  # 增加训练轮数和调整批量大小

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
