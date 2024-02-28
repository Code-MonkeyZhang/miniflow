
import sys
from pathlib import Path
# 将src目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import util
import data_processing as dp

if __name__ == '__main__':

    # load data
    x_train, y_train,z_train= dp.load_data()
    print(x_train)
    print(y_train)
    print(z_train)

    # set parameters
    initial_w = 0.0
    initial_b = 0.0
    learning_rate = 0.01
    num_iterations = 100

    # do simple linear regression
    w, b = util.simple_linear_regression(x_train, y_train,
        initial_w, initial_b, learning_rate, num_iterations)
    
