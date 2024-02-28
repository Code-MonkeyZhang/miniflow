
import sys
from pathlib import Path
# 将src目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import util
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # load data
    x_train, y_train= dp.load_data()

    # set parameters
    initial_w = 0.0
    initial_b = 0.0
    learning_rate = 0.01
    num_iterations = 500


    # do simple linear regression
    w_final, b_final = util.simple_linear_regression(x_train, y_train,
        initial_w, initial_b, learning_rate, num_iterations)
    
    # do prediction 
    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w_final * x_train[i] + b_final

    # plot the result
    util.plot_data_with_linear_fit(x_train, y_train, predicted)
