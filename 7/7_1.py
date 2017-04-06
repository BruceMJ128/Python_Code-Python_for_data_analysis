# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np

from pandas import DataFrame

from pandas import Series

df1 = DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
df2 = DataFrame({'key':['a','b','d'],'data2':range(3)})

df3 = pd.merge(df1,df2)

df4=pd.merge(df1, df2, how='outer')

df5=pd.merge(df1,df2,on='key',how='left')

left = DataFrame({'key1':['foo','foo','bar'], 'key2':['one','two','one'],'rval':[1,2,3]})

right = DataFrame({'key1':['foo','foo','bar','bar'], 'key2':['one','one','one','two'],'rval':[4,5,6,7]})
df6 = pd.merge(left, right, on=['key1','key2'], how='outer')

df7=pd.merge(left, right, on='key1')
df8 = pd.merge(left, right, on='key1', suffixes = ('_left','_right'))

left1 = DataFrame({'key':['a','b','a','a','b','c'],'value':range(6)})
right1 = DataFrame({'group_val':[3.5, 7]}, index = ['a','b'])

df8 = pd.merge(left1, right1, left_on = 'key',right_index = True, how='outer')


lefth = DataFrame({'key1':['Ohio','Ohio','Ohil','Nevada','Nevada'],
                    'key2':[2000,2001,2002,2001,2002],
                    'data':np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6,2)),
                    index = [['Nevada','Nevada','Ohio','Ohio','Ohio','Ohio'],
                    [2001,2000,2000,2000,2001,2002]],
                    columns=['event1','event2'])
df9 = pd.merge(lefth, righth, left_on = ['key1','key2'],right_index = True, how='outer')

left2 = DataFrame([[1.,2.],[3.,4.],[5.,6.]], index=['a','c','e'],columns=['Ohio','Nevada'])
right2 = DataFrame([[7.,8.],[9.,10.],[11.,12.],[13,14]], index=['b','c','d','e'],columns=['Missouri','Alabama'])

df10=pd.merge(left2, right2, how='outer',left_index=True, right_index=True)

df11 = left2.join(right2, how='outer')
df12 = left1.join(right1, on='key')

arr = np.arange(12).reshape((3,4))
arr2 = np.concatenate([arr,arr], axis=1)
arr3 = np.concatenate([arr,arr])

s1 = Series([0,1],index=['a','b'])
s2 = Series([2,3,4],index=['c','d','e'])
s3 = Series([5,6],index=['f','g'])

df13 = pd.concat([s1,s2,s3])

s4 = pd.concat([s1*5, s3])

df14= pd.concat([s1,s4], axis=1, join_axes=[['a','c','b','g']])

result = pd.concat([s1,s1,s3], keys = ['one','two','three']) #result type is Series
result2 = pd.concat([s1,s1,s3], keys = ['one','two','three'],axis=1) #result2 type is DataFrame

df15=result.unstack()

df21=DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'],columns=['one','two'])
df22=DataFrame(5+np.arange(4).reshape(2,2), index=['a','c'],columns=['three','four'])

df23 = pd.concat([df1,df2], axis=1, keys=['level1','level2']) #DataFrame相互合并

df24 = pd.concat({'level1':df1, 'level2':df2}, axis=1) #字典合并后转化为DataFrame，字典的key转化为DataFrame列头的key

df31 = DataFrame(np.random.randn(3,4), columns = ['a','b','c','d'])
df32 = DataFrame(np.random.randn(2,3), columns = ['b','d','a'])

df33 = pd.concat([df1,df2])
df34 = pd.concat([df1,df2], ignore_index=True)




A= Series([np.nan,2.5,np.nan, 3.5,4.5,np.nan], index=['f','e','d','a','b','c'])
B= Series(np.arange(len(A)), dtype = np.float64, index=['f','e','d','a','b','c'])


df41=np.where(pd.isnull(A),B,A) #A中的null元素，用B中对应的元素补上，其他则沿用A

df42=B[:-2].combine_first(A[2:]) #A和B合并，相同的元素用B，不同的部分，各自补上

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
result3 = data.stack()


df51 = DataFrame({'left':result3, 'right':result3+5},columns=pd.Index(['left','right'], name='side'))
df52 = df51.unstack('state')
df53 = df51.unstack('state').stack('side')




data = pd.read_csv('data/ch07/macrodata.csv')
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date') 
data1 = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))

ldata = data1.stack().reset_index().rename(columns={0: 'value'})
wdata = ldata.pivot('date', 'item', 'value')

ldata['value2']=np.random.randn(len(ldata))

pivoted = ldata.pivot('date','item')

data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
                  
data['v1'] = range(7)
data.drop_duplicates(['k1'])

                                    

data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}

data['animal'] = data['food'].map(str.lower).map(meat_to_animal) #map(str.lower)将所有字母都转化为小写，map(meat_to_animal)映射

data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
                
                
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})
            
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)

data = np.random.rand(20)
pd.cut(data, 4, precision=2)

cats = pd.qcut(data, 4) # Cut into quartiles 

np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.describe()

col=data[3]
col1 =   col[np.abs(col)>3] #Series选择和转化
data1 = data[(np.abs(data)>3).any(1)] #DataFrame选择和转化，每行只要1个元素满足条件np.abs(data)>3即可  .any(1)

data[np.abs(data)>3] = np.sign(data)*3 #.sign(data)元素转换为 1或-1


df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)

df2 = df.take(sampler)

bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)

draws = bag.take(sampler)

df_temp=np.random.permutation(len(df))
df_temp1=df_temp[:3]

df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': [11,12,13,14,15,16]})
df2=pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/ch02/movielens/movies.dat', sep='::', header=None,
                        names=mnames)
genre_iter = (set(x.split('|')) for x in movies.genres) #type is genetator
genres = sorted(set.union(*genre_iter)) 

dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

for i, gen in enumerate(movies.genres):
    dummies.ix[i, gen.split('|')] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]

np.random.seed(12345)
values = np.random.rand(10)

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
bins2=pd.get_dummies(pd.cut(values, bins))