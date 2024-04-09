import numpy as np

dL_dz_softmax = np.array([[-2.452723, -5.723021, -4.087872],
                          [2.452723, 5.723021, 4.087872]])

W_softmax = np.array([[0.1, 0.1, 0.1],
                      [0.2, 0.2, 0.2]])

dL_dz_relu = np.dot(dL_dz_softmax, W_softmax.T)

print(dL_dz_relu)