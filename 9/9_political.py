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

fec = pd.read_csv('data/ch09/P00000001-ALL.csv')

unique_cands = fec.cand_nm.unique()

parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
           
fec['party'] = fec.cand_nm.map(parties)

fec = fec[fec.contb_receipt_amt > 0]

fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]

fec.contbr_occupation.value_counts()[:10]

occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)

by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')
                                
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm

over_2mm.plot(kind='barh')

def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()    
    return totals.order(ascending=False)[:n] # Order totals by key in descending order
    
grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)

grouped.apply(get_top_amounts, 'contbr_employer', n=10)

bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins) #type is pandas.core.series.Series
labels

grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)

bucket_sums = grouped.contb_receipt_amt.sum().unstack(0) #axis = 0,按照横向展开
bucket_sums

normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
normed_sums

normed_sums[:-2].plot(kind='barh', stacked=True)

grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
totals[:10]

percent = totals.div(totals.sum(1), axis=0)
percent[:10]

from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import shapefile

obama = percent['Obama, Barack']

fig=plt.figure(figsize=(12,12))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

lllat = 21; urlat=53; lllon=-118; urlon=-62

m = Basemap(ax=ax, projection='stere',lon_0=(urlon + lllon)/2, lat_0=(urlat+lllat)/2, llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon, resolution = 'l')

m.drawcoastlines()
m.drawcountries()
#m.drawstates() #画出每个州的边界

sf = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020')
shp = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020.shp')
dbf = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020.dbf')

shapes=sf.shapes()
points=shapes[3].points

shpsegs=[]
for point in points:
    shpsegs.append(zip(point))
'''
lines = LineCollection(shpsegs, linewidths=(0.5, 1, 1.5, 2),linestyles='solid')

lines.set_facecolor('k')
lines.set_edgecolors('k')
lines.set_linewidth(0.3)
'''
'''
for npoly in range(shp.info()[0]):
    #在地图上绘制彩色多边形
    shpsegs = []
    shp_object = shp.read_object(npoly)
    verts = shp_object.vertives()
    rings = len(verts)
    for ring in range(rings):
        lons, lats = zip(*verts[ring])
        x, y = m(lons, lats)
        shpsegs.append(zip(x,y))
        if ring ==0:
            shapedict = dbf.read_record(npoly)
        name = shapedict['STATE']
    lines = LineCollection(shpsegs, antialiaseds=(1,))
    
    # state_to_code字典，例如'ALASKA' -> 'AK', omitted
    try:
        per =obama[state_to_code[name.upper()]]
    except KeyError:
        continue
    
    lines.set_facecolor('k')
    lines.set_alpha(0.75*per) #把“百分比”变小一点
    lines.set_edgecolors('k')
    lines.set_linewidth(0.3)
'''
plt.show()

