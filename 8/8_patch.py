# -*- coding: utf-8 -*-
from __future__ import division
from datetime import datetime
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd 

fig = plt.figure() #type of fig is matplotlib.figure.Figure
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3) #aplpha 反应颜色的深浅程度
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

plt.savefig('figpath.svg')
plt.savefig('figpath.png',dpi=400,bbox_inches='tight') #dpi 分辨率，bbox_inches 剪除图表中的空白部分

plt.savefig('figpath_2.png') #dpi 分辨率，bbox_inches 剪除图表中的空白部分

from io import BytesIO
buffer = BytesIO() #type is <_io.BytesIO at 0xd376db0>
plt.savefig(buffer)  
plot_data = buffer.getvalue() #type of plot_data is str,用于在Web上动态生成图片

plt.rc('figure', figsize=(10, 10)) #设置图片尺寸为10*10
plt.savefig('figpath_2.png')

plt.close('all')

s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10)) #index，x轴范围0~100,间隔10
#figure_s=s.plot() #type of figure_s is matplotlib.axes._subplots.AxesSubplot
#s.value_counts().plot(kind='bar')

df = DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))

fig, axes = plt.subplots(2, 1)
data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
#data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
fig3 = data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)

df = DataFrame(np.random.rand(6, 4),
               index=['one', 'two', 'three', 'four', 'five', 'six'],
               columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))

#df.plot(kind='barh', stacked=True)

tips = pd.read_csv('data/ch08/tips.csv') #type of tips is DataFrame
party_counts = pd.crosstab(tips.day, tips.sizes) #type of tips is DataFrame

# Not many 1- and 6-person parties
#party_counts = party_counts.ix[:, 2:5]

party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0) #归一化，party_counts.sum(1)沿着axis=1轴取和

#party_pcts.plot(kind='bar', stacked=True)

tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins=50) #type is matplotlib.axes._subplots.AxesSubplot, Series.hist()画出直方图

tips['tip_pct'].plot(kind='kde') #kind='kde'生产密度图，标准混合正态分布KDE, kind='kde'的目的是为了将图像做平滑处理，防止过拟合

comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)中心值为0，标准差为 1, type is numpy.ndarray 
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True) #直方图，bins直方图中各bin的顶点位置
values.plot(kind='kde', style='k--')

macro = pd.read_csv('data/ch08/macrodata.csv') #type is pandas.core.frame.DataFrame
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']] #type is pandas.core.frame.DataFrame
trans_data = np.log(data).diff().dropna() #.diff()向下平移并与平移前的数组求差
trans_data[-5:]

plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))

pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3) #diagonal，相当于是自己的横坐标与数量的撒点，拟合为正态分布

