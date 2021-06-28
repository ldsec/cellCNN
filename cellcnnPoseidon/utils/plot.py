import numpy as np
# import matplotlib.pyplot as plt

M = np.array([[0,1,2], [3,4,5]])
MT = M.T

v = M.flatten()
vt = MT.flatten()
transposeMatrix = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1],
     ]
)
res = v @ transposeMatrix
print("\n######## Example of Transpose ########\n")
print("=> Original matrix M(2x3):\n{}".format(M))
print("=> Transpose matrix M.T(3x2):\n{}".format(MT))
print("=> Row packing of M, vector v:\n{}".format(v))
print("=> Row packing of M.T, vector vt:\n{}".format(vt))
print("=> Transformation Matrix Q(6x6):\n{}".format(transposeMatrix))
print("=> v * Q = vt")