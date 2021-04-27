import numpy as np
# import matplotlib.pyplot as plt

# ATT_LSTM = [2100,0,79]
# #x = ['REST','LAPT','AUTO']
# x = np.arange(3) #总共有几组，就设置成几，我们这里有三组，所以设置为3
# total_width, n = 0.8, 1    # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
# width = total_width / n
# x = x - (total_width - width) / 2
# plt.bar(x, ATT_LSTM, color = "r",width=width,label='ATT-LSTM ')
# # plt.bar(x + width, MATT_CNN, color = "y",width=width,label='MATT-CNN')
# # plt.bar(x + 2 * width, ATT_RLSTM , color = "c",width=width,label='ATT-RLSTM')
# # plt.bar(x + 3 * width, CNN_RLSTM , color = "g",width=width,label='CNN-RLSTM')
# plt.xlabel("Layers")
# plt.ylabel("Milliseconds")
# plt.legend(loc = "best")
# plt.xticks([0,1,2],['Conv1D','Pooling','Dense'])
# plt.yscale("log")
# # my_y_ticks = np.arange(0.8, 0.95, 0.02)
# # plt.ylim((0.8, 0.95))
# # plt.yticks(my_y_ticks)
# plt.show()

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