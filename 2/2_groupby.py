import pandas as pd; import numpy as np; import pylab as pl
df = pd.DataFrame({'key1':['a','a','b','b','a'], 'key2':['one','two','one','two','one'],'data1':np.random.randn(5),'data2':np.random.randn(5)})

grouped = df['data1'].groupby(df['key1'])

means = df['data1'].groupby([df['key1'], df['key2']]).mean()

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
grouped2=df['data1'].groupby([states, years])

grouped3=df.groupby('key1')

grouped4=df.groupby(['key1', 'key2'])

group_size =  df.groupby(['key1', 'key2']).size()

pieces = dict(list(df.groupby('key1')))
pieces2= dict(list(df.groupby(['key1', 'key2'])))

