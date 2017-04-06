# -*- coding: utf-8 -*-
from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd

np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))

np.set_printoptions(precision=4)

fig = plt.figure() #在输入框中输入fig,图像会弹出
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)


plt.plot(randn(50).cumsum(),'k--')

_ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))

plt.close('all') #用于关闭图形，结束对之前的操作

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')
ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')


fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'k', label='_nolegend_')#label='_nolegend_' 不显示图例
ax.plot(randn(1000).cumsum(), 'r--', label='two')
ax.plot(randn(1000).cumsum(), 'g.', label='three')

ax.legend(loc='best')

