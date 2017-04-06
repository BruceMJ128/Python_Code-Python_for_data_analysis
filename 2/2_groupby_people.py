# -*- coding: utf-8 -*-
from pandas import DataFrame, Series
import pandas as pd; import numpy as np; import pylab as pl

people = pd.DataFrame(np.random.randn(5,4), columns = ['a','b','c','d'], index=['Joe','Steve','Wes','Jim','Travis'])

people.ix[2:3,['b','c']]=np.nan #change Wes's b and c to NaN

mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}

by_column = people.groupby(mapping, axis=1) #mapping这个位置可以是dict，也可以是series

map_series = pd.Series(mapping)

map_series_gb=people.groupby(map_series, axis=1)

columns2 = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],[1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4,5),columns=columns2)

