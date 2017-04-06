# -*- coding: utf-8 -*-
from __future__ import division
from datetime import datetime
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('data/ch08/spx.csv', index_col=0, parse_dates=True) #type of data is DataFrame
spx = data['SPX']

spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
                xytext=(date, spx.asof(date) + 200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left', verticalalignment='top')
                
                #xy箭头终点离点的高度，xytext箭头起点和文字距点的高度
#spx.asof(date)，对于spx中对应的date，得到高度值

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2012'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in 2008-2009 financial crisis')