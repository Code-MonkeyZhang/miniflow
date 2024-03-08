
from matplotlib import pyplot as plt
import numpy as np
import copy
import math

"""
======================================================================
Linear Regression
======================================================================
"""


# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        predict = np.dot(w, x[i]) + b
        cost = cost + (predict - y[i]) ** 2
    total_cost = cost / (2 * m)

    return total_cost


def compute_gradient(x_train, y_train, w, b):
    # m = number of training examples
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        # compute the prediction of current w,b
        y_predict = w * x_train[i] + b
        dj_dw_i = (y_predict - y_train[i]) * x_train[i]
        dj_db_i = (y_predict - y_train[i])

        # sum all m parameters
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# gradient descent
def simple_linear_regression(x_train, y_train, w_init, b_init, alpha, num_iterations):
    """
    Perform simple linear regression.

    参数:
    - x_train: numpy array, 训练数据集的特征值/输入变量。
    - y_train: numpy array, 训练数据集的目标值/输出变量。
    - w_init: float, 权重的初始值。
    - b_init: float, 偏置的初始值。
    - alpha: float, 学习率，用于控制优化步骤的大小。
    - num_iterations: int, 梯度下降算法的迭代次数。

    返回:
    - w: float, 优化后的权重。
    - b: float, 优化后的偏置。

    函数执行简单线性回归，使用梯度下降法优化权重和偏置。
    """

    # initialize w and b
    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init)

    # compute gradient
    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        # update w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 10 == 0:
            print('iteration {}: w = {}, b = {}'.format(i, w, b))
    return w, b


def plot_data_with_linear_fit(x_data, y_data, predicted, scatter_color='r', line_color='b', marker='x'):
    """
    绘制数据点的散点图和线性拟合线。

    参数:
    - x_data: x轴数据点。
    - y_data: y轴数据点。
    - predicted: 预测值，用于线性拟合线。
    - x_label: x轴标签。
    - y_label: y轴标签。
    - title: 图表标题。
    - scatter_color: 散点图的颜色，默认为红色。
    - line_color: 线性拟合线的颜色，默认为蓝色。
    - marker: 散点图的标记类型，默认为'x'。
    """
    # 绘制线性拟合线
    plt.plot(x_data, predicted, c=line_color)
    # 创建散点图
    plt.scatter(x_data, y_data, marker=marker, c=scatter_color)
    # 显示图形
    plt.show()


def compute_linear_gradient(x_train, y_train, w, b):
    size = x_train.shape[0]
    features = x_train.shape[1]
    dj_dw = np.zeros(features)
    dj_db = 0

    # 遍历每一行数据
    for i in range(size):
        error = (np.dot(x_train[i], w) + b - y_train[i])
        # 遍历每一个feature
        for j in range(features):
            dj_dw[j] += error * x_train[i, j]  # 计算每个feature的导数，然后累加起来，方便后面求平均
        # 对于 b, 只需要算一次
        dj_db += error

    dj_dw = dj_dw / size
    dj_db = dj_db / size

    return dj_dw, dj_db


def multi_feature_linear_regression(x_train, y_train, w_init, b_init, alpha, num_iterations):
    """
    Perform multi-feature linear regression.
    x_train has multiple features
    """

    # check if the vectors are aligned
    if x_train.shape[1] != w_init.shape[0]:
        print("the column of w and the size of x_train do not match!")
        return -1

    # init variables
    w = copy.deepcopy(w_init)
    b = b_init
    cost_history = []

    # runing gradient decent
    for i in range(num_iterations):
        # compute gradient for w and b
        dj_dw, dj_db = compute_linear_gradient(x_train, y_train, w, b)

        # update w[] and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # compute cost for this pair of (w,b)
        cost_history.append(compute_cost(x_train, y_train, w, b))

        if i % 10 == 0:
            print(
                f"Iteration {i:4d}: W {w},B{b}, Cost {cost_history[-1]:8.2f}")

    return w, b


# feature scaling
def feature_scaling(data: np.ndarray, type: str) -> np.ndarray:
    if type == 'z-score':
        # Z-Score Normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
    elif type == 'min-max':
        # Min-Max Normalization
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif type == 'mean':
        # Mean Normalization
        mean = np.mean(data, axis=0)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - mean) / (max_val - min_val)
    else:
        raise ValueError("Type must be 'z-score', 'min-max', or 'mean'")

    return normalized_data