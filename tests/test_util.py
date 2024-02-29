
import sys
from pathlib import Path
# 将src目录添加到sys.path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import util
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 测试util模块
    # 1. 测试util中的函数
    # 1.1 测试load_data函数
    # 1.1.1 测试load_data函数的返回值
    print("hello")