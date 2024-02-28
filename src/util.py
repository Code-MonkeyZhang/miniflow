from matplotlib import pyplot as plt
import numpy as np
import copy
import math

# Function to calculate the cost


# Function to calculate the cost
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x_train, y_train, w, b):
    # m = number of training examples
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        # compute the prediction of current w,b
        y_predict = w*x_train[i]+b
        dj_dw_i = (y_predict-y_train[i])*x_train[i]
        dj_db_i = (y_predict-y_train[i])

        # sum all m parameters
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

# feature scaling
def feature_scaling(x_train):
    return x_train


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
    J_history = []
    w_history = []

    # initialize w and b
    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init)

    # compute gradient

    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        # update w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
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