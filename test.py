import numpy as np
from src.miniflow.util import compute_cross_entropy_loss

# 示例数据
prediction = np.array([
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9],  # 错误地非常确信最后一个类
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 预测分布均匀，无确信信息
     [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]  # 高度确信第一个类，且正确
)

target = np.array(

[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 真实类别是第一类
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 真实类别是第八类
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 真实类别是第一类

)

# 计算损失
loss = compute_cross_entropy_loss(prediction, target)
print("Computed cross-entropy loss:", loss)
