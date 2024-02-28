import sys
import numpy as np
from pathlib import Path

# 将src目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from util import *

def load_data():
    # load data
    data=np.loadtxt('data/simple_gd_data.txt', delimiter=',')
    x_train = data[:,0]
    y_train = data[:,1]
    return x_train, y_train