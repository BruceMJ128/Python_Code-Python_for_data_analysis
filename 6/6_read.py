import pandas as pd; import numpy as np; import pylab as pl

from pandas import Series

df = pd.read_csv('data/ch06/ex1.csv')

df2 = pd.read_table('data/ch06/ex1.csv', sep=',')

names1 = ['a','b','c','d','message']
df3=pd.read_csv('data/ch06/ex2.csv', names=names1, index_col='message')

df4 = list(open('data/ch06/ex3.txt'))
df5 = pd.read_csv('data/ch06/ex3.txt')

result = pd.read_table('data/ch06/ex3.txt', sep='\s+')

df6 = list(open('data/ch06/ex4.csv'))
df7 = pd.read_csv('data/ch06/ex4.csv', skiprows=[0,2,3])


df8 = pd.read_csv('data/ch06/ex5.csv')
sen = {'message':['foo','NA'],'something':['two']}
df9=pd.read_csv('data/ch06/ex5.csv', na_values=sen)

df10_list = list(open('data/ch06/ex6.csv'))
df10 = pd.read_csv('data/ch06/ex6.csv', nrows=5)

chunker = pd.read_csv('data/ch06/ex6.csv', chunksize = 1000)

tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
    
#tot = tot.order(ascending = False)

