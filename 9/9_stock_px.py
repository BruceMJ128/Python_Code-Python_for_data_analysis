# -*- coding: utf-8 -*-
from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

x1 = list(np.arange(10))
x2 = list(np.arange(10)*3+5)

x3 = pd.Series(np.random.randn(10))

dict1 = {'x1':x1, 'x2':x2}
df = pd.DataFrame(dict1)

y1 = df['x1'].corr(df['x2'])

y2 = df.corr()['x2']

y3 = df.corrwith(df['x2'])

y4 = df.corrwith(x3)

close_px = pd.read_csv('data/ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px.info()

rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX']) #DataFrame.corrwith(Series) 求出相关系数
by_year = rets.groupby(lambda x: x.year)
by_year.apply(spx_corr) #求出两个Series之间的相关系数，再组合为DataFrame

# Annual correlation of Apple with Microsoft
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit() #线性拟合 y = ax + b
    return result.params #将a和b作为返回值

by_year.apply(regress, 'AAPL', ['SPX']) #返回值，SPX一列即y = ax + b中的a，intercept即截距b

'''
output:
           SPX  intercept
2003  1.195406   0.000710
2004  1.363463   0.004201
2005  1.766415   0.003246
2006  1.645496   0.000080
2007  1.198761   0.003438
2008  0.968016  -0.001110
2009  0.879103   0.002954
2010  1.052608   0.001261
2011  0.806605   0.001514    
'''