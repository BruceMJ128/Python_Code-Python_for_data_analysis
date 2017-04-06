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

tips = pd.read_csv('data/ch08/tips.csv')
# Add tip percentage of total bill
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]

grouped = tips.groupby(['sex','smoker'])

grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')


grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)

grouped_pct.agg(['mean', 'std', peak_to_peak])

functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)

ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)

#grouped.agg({'tip':np.max,'size':'sum'})
dict1 = {'tip_pct' : ['min', 'max', 'mean', 'std'],'sizes':'sum'}
grouped.agg(dict1)

def top(df, n=5, column='tip_pct'):
    return df.sort_index(by=column)[-n:]
top(tips, n=6)

tips.groupby('smoker').apply(top)

tips.groupby('smoker').apply(top, n=1, column='total_bill')

tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')

result = tips.groupby('smoker')['tip_pct'].describe()
result

result.unstack('smoker')

f = lambda x: x.describe()
grouped.apply(f)

tips.pivot_table(index=['sex', 'smoker'])

tips.pivot_table(['tip_pct', 'sizes'], index=['sex', 'day'],columns='smoker')
                 
tips.pivot_table(['tip_pct', 'sizes'], index=['sex', 'day'],columns='smoker', margins=True)

tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day',aggfunc=len, margins=True)

tips.pivot_table('sizes', index=['time', 'sex', 'smoker'],columns='day', aggfunc='sum', fill_value=0)

from StringIO import StringIO
data = """\
Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')

pd.crosstab(data.Gender, data.Handedness, margins=True)

pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)