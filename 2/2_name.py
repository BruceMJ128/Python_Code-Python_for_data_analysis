# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np; import pylab as pl

#read all data in txt from file

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'data/ch02/names/yob%d.txt' %year  
    frame = pd.read_csv(path, names = columns) #读取所有txt文件
    
    frame['year'] = year
    pieces.append(frame)  #pieces is list
    
names = pd.concat(pieces, ignore_index = True) #将list整合到dataframe中

total_births = names.pivot_table('births', index='year', columns = 'sex', aggfunc = sum) #数据处理

def add_prop(group):
    births = group.births.astype(float)    #转化为小数
    group['prop']=births / births.sum()   #添加一列比例数据
    return group
names = names.groupby(['year','sex']).apply(add_prop)  

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]

grouped = names.groupby(['year','sex'])
top1000 = grouped.apply(get_top1000)

#analyze naming trendency

boys = top1000[top1000.sex=='M']

girls = top1000[top1000.sex=='F']

# analysis of last letter

get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table= names.pivot_table('births', index=last_letters, columns = ['sex','year'],aggfunc=sum)

subtable = table.reindex(columns = [1910,1960,2010], level='year')

letter_prop = subtable / subtable.sum().astype(float)

#import matplotlib.pyplot as plt
#fig, axes = plt.subplots(2,1,figsize=(10,8))
#letter_prop['M'].plot(kind = 'bar', rot=0, ax=axes[0], title = 'Male')
#letter_prop['F'].plot(kind = 'bar', rot=0, ax=axes[1], title = 'Female', legend = False)

letter_prop2 = table/ table.sum().astype(float)
dny_ts = letter_prop2.ix[['d','n','y'],'M'].T

#boys' name transform to girl's name

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]

filtered = top1000[top1000.name.isin(lesley_like)]

table = filtered.pivot_table('births', index='year', columns = 'sex', aggfunc='sum')
#table1 = table.div(table.sum(1), axis=0)