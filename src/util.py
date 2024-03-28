from matplotlib import pyplot as plt
import numpy as np
import copy
import math
from typing import List, Callable
from src.activation import *



def regression(x_train, y_train, w_init, b_init, alpha, num_iterations, mode, lambda_=0):
    print(f"Performing {mode} regression")
    """
    Perform multi-feature regression.
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

    # running gradient decent
    for i in range(num_iterations):
        # compute gradient for w and b
        if mode == 'linear':
            dj_dw, dj_db = compute_linear_gradient(
                x_train, y_train, w, b, lambda_)
        elif mode == 'logistic':
            dj_dw, dj_db = compute_log_gradient(
                x_train, y_train, w, b, lambda_)

        # update w[] and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # compute cost for this pair of (w,b)
        cost_history.append(compute_linear_cost(x_train, y_train, w, b))

        if i % 10 == 0:
            print(
                f"Iteration {i:4d}: W {w},B{b}, Cost {cost_history[-1]:8.2f}")

    return w, b


"""
======================================================================
Linear Regression
======================================================================
"""


# Function to calculate the cost
def compute_linear_cost(x, y, w, b, lambda_=0):
    m = x.shape[0]
    n = x.shape[1]
    cost = 0

    for i in range(m):
        predict = np.dot(w, x[i]) + b
        cost = cost + (predict - y[i]) ** 2
    total_cost = cost / (2 * m)

    # add regularization term
    for i in range(n):
        total_cost += (lambda_ / (2 * m)) * w[i] ** 2
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


def compute_linear_gradient(x_train, y_train, w, b, lambda_=0):
    size = x_train.shape[0]
    features = x_train.shape[1]
    dj_dw = np.zeros(features)
    dj_db = 0

    # 遍历每一行数据
    for i in range(size):
        f_wb = linear_function(x_train[i], w, b)
        error = (f_wb - y_train[i])
        # 遍历每一个feature
        for j in range(features):
            dj_dw[j] += error * x_train[i, j]  # 计算每个feature的导数，然后累加起来，方便后面求平均
        # 对于 b, 只需要算一次
        dj_db += error

    dj_dw = dj_dw / size
    dj_db = dj_db / size

    for j in range(features):
        dj_dw[j] += lambda_ * w[j] / size

    return dj_dw, dj_db


"""
======================================================================
Logistic Regression
======================================================================
"""


def log_1pexp(x, maximum=20):
    ''' approximate log(1+exp^x)
    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    '''

    out = np.zeros_like(x, dtype=float)
    i = x <= maximum
    ni = np.logical_not(i)

    out[i] = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_log_cost(x_train, y_train, w, b, lambda_=0, safe=False):
    """
    Computes cost using logistic loss, non-matrix version

    Args:
      X (ndarray): Shape (m,n)  matrix of examples with n features
      y (ndarray): Shape (m,)   target values
      w (ndarray): Shape (n,)   parameters for prediction
      b (scalar):               parameter  for prediction
      lambda_ : (scalar, float) Controls amount of regularization, 0 = no regularization
      safe : (boolean)          True-selects under/overflow safe algorithm
    Returns:
      cost (scalar): cost
    """

    m, n = x_train.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x_train[i], w) + b  # (n,)(n,) or (n,) ()
        if safe:  # avoids overflows
            cost += -(y_train[i] * z_i) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid_function(z_i)  # (n,)
            # scalar
            cost += -y_train[i] * np.log(f_wb_i) - \
                (1 - y_train[i]) * np.log(1 - f_wb_i)
    cost = cost / m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            # scalar
            reg_cost += (w[j] ** 2)
        reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost


def compute_log_gradient(x_train, y_train, w, b, lambda_=0):
    """
    Computes the gradient for logistic regression 

    Args:
      x_train (ndarray (size,features): Data, m examples with n features
      y_train (ndarray (m,)): target values
      w (ndarray (features,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (features,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    size = x_train.shape[0]
    features = x_train.shape[1]
    dj_dw = np.zeros(features)
    dj_db = 0.0  # logistic function, b is a floating point number

    for i in range(size):
        # x_train[i] i_th row of data
        f_wb = sigmoid_function(np.dot(x_train[i], w) + b)
        error = f_wb - y_train[i]
        for j in range(features):
            # each features multiply error as the formula (f_wb-y[i])*x[i,j]
            dj_dw[j] += error * x_train[i, j]
        dj_db += error

    dj_dw = dj_dw / size
    dj_db = dj_db / size

    for j in range(features):
        dj_dw[j] += lambda_ * w[j] / size
    return dj_dw, dj_db


def plot_decision_boundary(X, y, w, b):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 1000),
                           np.linspace(x2_min, x2_max, 1000))

    # 根据 w 和 b 计算模型的预测
    Z = w[0] * xx1 + w[1] * xx2 + b
    Z = 1 / (1 + np.exp(-Z))  # 应用sigmoid函数
    Z = Z >= 0.5  # 转换为类别

    # 绘制分类边界
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Custom Logistic Regression Decision Boundary')


"""

Dense Layer
======================================================================
"""

#
# def myDense(units, a_in: np.ndarray, W: np.ndarray, b: np.ndarray, active_func: Callable) -> np.ndarray:
#     # init variables
#     units = a_in.shape[1]
#     a_out = np.zeros(units)
#     # iterate and compute each node
#     for i in range(units):
#         w = W[:, i]
#         a = np.dot(w, a_in) + b[i]
#         a_out[i] = active_func(a)
#     return a_out
#
#
# def mySequential(x, W, b):
#     a1 = myDense(x, W[0], b[0], "sigmoid")
#     a2 = myDense(a1, W[1], b[1], "sigmoid")
#     a3 = myDense(a2, W[2], b[2], "sigmoid")
#     final_output = a3
#     return final_output
#

'''
Use Case:
model = Sequential(
    [               
        Dense(25, activation="sigmoid", name="layer1"),
        Dense(15, activation="sigmoid", name="layer2"),
        Dense(1, activation="sigmoid", name="layer3"),
    ], name = "my_model" )
'''


class Dense:
    """
    Dense layer Class
    """

    def __init__(self, units: int, activation: str, name='layer'):
        # init variables
        self.units = units
        self.activation = activation
        self.name = name
        self.Weights = None
        self.Biases = None

    def compute_layer(self, a_in: np.ndarray) -> np.ndarray:
        # init variables
        # iterate and compute each node
        a = np.dot(a_in, self.Weights) + self.Biases
        if self.activation == "sigmoid":
            a_out = sigmoid_function(a_in)
        return a

    @staticmethod
    def sigmoid_function(z):
        g = 1 / (1 + np.exp(-z))
        return g

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        self.Weights = w
        self.Biases = b


class Model:
    def __init__(self, dense_array: List[Dense], name='model') -> None:
        self.dense_array = dense_array
        self.name = name

    def predict(self, x: np.ndarray) -> np.ndarray:
        for dense_layer in self.dense_array:
            x = dense_layer.compute_layer(x)
        return x

    # TBD
    # def fit(self, X_train, y_train, epochs):
    #     # perform backward prop
    #     pass
    # def evaluate(self):
    #     pass
    # def get_weights(self):
    #     pass



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
