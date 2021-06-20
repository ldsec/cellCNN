 # -*- coding: utf-8 -*-
import re
import os
import numpy as np
from matplotlib import pyplot as plt

# from pylab import rcParams
plt.style.use('seaborn')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 25}

plt.rc('font', **font)

def plot_micro():
     # 1. filter num
     x = np.array([2, 4, 8])
     
     # 2. data of batch = 1
     m1 = {
          'prepare': np.array([24.42755068, 25.19794894, 26.31910534]),
          'forward': np.array([1.75561309, 3.27513337, 6.40319527]),
          'backward': np.array([2.91261827, 4.86219892, 9.04591785]),
          'reduce': np.array([0.31215145, 0.50676524, 0.92202973]),
     }  
     s1 = {
          'prepare': np.array([0.8233933404, 0.8448166989, 0.9520308814]),
          'forward': np.array([0.01732030717, 0.03208901755, 0.1915892027]),
          'backward': np.array([0.02882096484, 0.04399822382, 0.2122780041]),
          'reduce': np.array([0.01097614845, 0.0195877831, 0.03575286591]),
     }  

     m10 = {
          'prepare': np.array([24.42755068, 25.19794894, 26.31910534]),
          'forward': np.array([19.26150172, 36.50775283, 70.09024712]),
          'backward': np.array([29.0742886, 49.00082651, 88.17362271]),
          'reduce': np.array([0.33620251, 0.55323338, 0.95894135]),
     }  
     s10 = {
          'prepare': np.array([0.8233933404, 0.8448166989, 0.9520308814]),
          'forward': np.array([0.1852565863, 0.8064517305, 2.047155328]),
          'backward': np.array([0.2570277014, 0.8772146653, 2.883161479]),
          'reduce': np.array([0.009981389603, 0.02476735557, 0.03921506766]),
     }
     
     # fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(2, 3, sharey=True)
     fig, ax_list = plt.subplots(1, 2, figsize=(20, 5), sharey=False)
     ax_list = ax_list.flatten()
     ax_list[0].set_ylim([0, 40])
     
     for phaze in m1.keys():
          ax_list[0].errorbar(x, m1[phaze], yerr=s1[phaze], fmt='-o',
                    mec='yellow', ms=5, mew=1, label=phaze)
     ax_list[0].set_title('Batch Size = 1', fontweight='bold')
     
     for phaze in m10.keys():
          ax_list[1].errorbar(x, m10[phaze], yerr=s10[phaze], fmt='-o',
                    mec='yellow', ms=5, mew=1, label=phaze)
     ax_list[1].set_title('Batch Size = 10', fontweight='bold')
     
     for ax in ax_list:
          ax.set_xlabel("Number of filters", fontweight='bold')
          ax.legend(loc='upper left')

     ax_list[0].set_ylabel("Overhead(s)", fontweight='bold')
     
     
     
     plt.savefig("micro.pdf")
     plt.show()


if __name__ == "__main__":
    plot_micro()
