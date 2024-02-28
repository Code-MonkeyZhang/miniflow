import sys
import numpy as np
from pathlib import Path

# 将src目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from util import *

def load_data():
    # load data
    data=np.loadtxt('data/simple_gd_data.txt', delimiter=',')
    print(data)
    print(data.shape)
    x_train = data[:,0]
    y_train = data[:,1]
    z_train = data[:,2]
    return x_train, y_train,z_train