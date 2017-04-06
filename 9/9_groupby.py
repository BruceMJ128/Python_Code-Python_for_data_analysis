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

pd.options.display.notebook_repr_html = False

df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})

grouped = df['data1'].groupby(df['key1'])
means = df['data1'].groupby([df['key1'], df['key2']]).mean()

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()

for name, group in df.groupby('key1'):
    print(name)
    print(group)
    
pieces = dict(list(df.groupby('key1')))

people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan # Add a few NA values
people

mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}
           
by_column = people.groupby(mapping, axis=1)

map_series = Series(mapping)
map_series

columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
#type of columns is pandas.indexes.multi.MultiIndex
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
hier_df

hier_df.groupby(level='cty', axis=1).count()

grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)

frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4)
factor[:10]

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()

#ADAPT the output is not sorted in the book while this is the case now (swap first two lines)

# Return quantile numbers
grouping = pd.qcut(frame.data1, 10, labels=False)

grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()

grouping2 = pd.qcut(frame.data1, 10)

grouped2 = frame.data2.groupby(grouping2)
grouped2.apply(get_stats).unstack()