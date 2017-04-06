import pandas as pd; import numpy as np
from pandas import Series, DataFrame
from numpy import nan

sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}

obj3 = Series(sdata)

states = ['CA', 'Ohio','Oregon', 'Texas']

obj4 = Series(sdata, index=states)


obj = Series(range(3), index = ['a','b','c'])

index = obj.index

index = pd.Index(np.arange(3))

obj2 = Series([1.5,-2.5,0], index = index)

obj = Series([4.5,7.2,-5.3,3.6], index = ['d','b','a','c'])
obj5 = obj.reindex(['a','b','c','d','e'],fill_value=0)

obj6 = Series(['blue','purple','yellow'], index=[0,2,4])
obj7 = obj6.reindex(range(6), method = 'ffill')

frame = DataFrame(np.arange(9).reshape((3,3)), index = ['a','c','d'], columns = ['Ohio','Texas','California'])

frame.index.name = "name"
frame.columns.name='State'

data = DataFrame(np.arange(16).reshape((4,4)), index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])

data1 = data.drop(['Colorado', 'Ohio'])

data2 = data.drop(['two', 'four'], axis=1)

data3 = data.ix['Colorado',['two','three']]
data4 = data.ix[['Colorado','Utah'],[3,0,1]]
data5 = data.ix[data.three>5, :3]

df1 = DataFrame(np.arange(12).reshape((3,4)), columns = list('abcd'))
df2 = DataFrame(np.arange(20).reshape((4,5)), columns = list('abcde'))

sum1 = df1+df2
sum2 = df1.add(df2, fill_value=0)

arr = np.arange(12).reshape((3,4))

arr_minus=arr-arr[0]

frame = DataFrame(np.arange(12).reshape((4,3)), columns = list('bde'), index = ['Utah','Ohio','Texas','Oregon'])
series3 = frame['d']
series2 = Series(range(3), index=['b','e','f'])

f_sub = frame.sub(series3, axis=0)
f_sub2 = frame-series2

f = lambda x: x.max()- x.min()
g = lambda x: '%.2f' %x

f1 = frame.apply(f)
f2 = frame.apply(f, axis=1)

f3=frame.applymap(g)

obj_rank = Series([7, -5,7,4,2,0,4])

o1 = obj_rank.rank()
o2 = obj_rank.rank(method = 'first')
o3 = obj_rank.rank(ascending=False, method = 'max')

df = DataFrame(np.random.randn(7,3))
df.ix[:4,1]= nan; df.ix[:2,2]= nan

df_dropna = df.dropna(thresh=2)

data = Series(np.random.randn(10), index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])

DF_data = data.unstack()
Se_data = DF_data.stack()

frame = DataFrame(np.arange(12).reshape((4,3)), index = [['a','a','b','b'],[1,2,1,2]], columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
