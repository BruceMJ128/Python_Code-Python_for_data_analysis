# -*- coding: utf-8 -*-
from __future__ import division
from datetime import datetime
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd 

data = pd.read_csv('data/ch08/Haiti.csv')
data.info()

data[['INCIDENT DATE', 'LATITUDE', 'LONGITUDE']][:10]

data.describe() #对于data中所有元素为数字的部分做统计和计算

data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) & (data.LONGITUDE < -70)
            & data.CATEGORY.notnull()]
            
def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(',')) #type is generator, 实质有点类似于list
    return [x for x in stripped if x]

'''
输入：
cat = data.CATEGORY[10]
Out[20]: "1a. Highly vulnerable, 2. Urgences logistiques | Vital Lines, 2d. Refuge | Shelter needed, 2a. Penurie d'aliments | Food Shortage, "
输出：
x = to_cat_list(cat)
x
Out[23]: 
['1a. Highly vulnerable',
 '2. Urgences logistiques | Vital Lines',
 '2d. Refuge | Shelter needed',
 "2a. Penurie d'aliments | Food Shortage"]
'''

def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series) #type is generator
    return sorted(set.union(*cat_sets))   #对于generator解引用，使用*

def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split(' | ')[1]
    return code, names.strip()
    
str1 = '2. Urgences logistiques | Vital Lines'
get_english(str1)

all_cats = get_all_categories(data.CATEGORY)
# Generator expression
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping['2a']
english_mapping['6c']

def get_code(seq):
    return [x.split('.')[0] for x in seq if x]

all_codes = get_code(all_cats)
code_index = pd.Index(np.unique(all_codes))
dummy_frame = DataFrame(np.zeros((len(data), len(code_index))),
                        index=data.index, columns=code_index)
                        
for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.ix[row, codes] = 1

data = data.join(dummy_frame.add_prefix('category_'))

from mpl_toolkits.basemap import Basemap #牛人做好了一个地球仪的包，直接调用即可
import matplotlib.pyplot as plt

def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25,lllon=-75, urlon=-71):
    # create polar stereographic Basemap instance.
    m = Basemap(ax=ax, projection='stere', lon_0=(urlon + lllon) / 2, lat_0=(urlat + lllat) / 2, llcrnrlat=lllat, urcrnrlat=urlat,llcrnrlon=lllon, urcrnrlon=urlon, resolution='f')
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

for code, ax in zip(to_plot, axes.flat):
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,lllon=lllon, urlon=urlon)

    cat_data = data[data['category_%s' % code] == 1]
    

    # compute map proj coordinates.
    x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)

    m.plot(x, y, 'k.', alpha=0.5)
    ax.set_title('%s: %s' % (code, english_mapping[code]))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

def make_plot():

    for i, code in enumerate(to_plot):
        cat_data = data[data['category_%s' % code] == 1]
        lons, lats = cat_data.LONGITUDE, cat_data.LATITUDE

        ax = axes.flat[i]
        m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,
                            lllon=lllon, urlon=urlon)

        # compute map proj coordinates.
        x, y = m(lons.values, lats.values)

        m.plot(x, y, 'k.', alpha=0.5)
        ax.set_title('%s: %s' % (code, english_mapping[code]))

shapefile_path = 'data/ch08/PortAuPrince_Roads/PortAuPrince_Roads'
m.readshapefile(shapefile_path, 'roads')

make_plot()
#------------------------------------#

######画出海地街道图#########

shapefilepath = 'data/ch08/PortAuPrince_Roads/PortAuPrince_Roads'

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

lat0 = 18.533333;lon0 = -72.333333;change = 0.13;
lllat=lat0-change; urlat=lat0+change; lllon=lon0-change; urlon=lon0+change;

m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,lllon=lllon, urlon=urlon)

m.readshapefile(shapefilepath,'roads') #添加街道数据

code = '2a'
cat_data = data[data['category_%s' % code] == 1]

# compute map proj coordinates.
x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)

m.plot(x, y, 'k.', alpha=0.5)
ax.set_title('Food shortages reported in Port-au-Prince')
plt.savefig('myfig.png',dpi=400,bbox_inches='tight')