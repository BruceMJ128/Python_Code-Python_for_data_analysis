# -*- coding: utf-8 -*-

import json
from pandas import DataFrame, Series
import pandas as pd; import numpy as np; import pylab as pl

db = json.load(open('data/ch07/foods-2011-10-03.json')) #type of db is list
len(db)

dict1 = db[0] #type of db[0] is dict

#db[0] structure shown below:

'''
value of 'description','group','id',''manufacturer','tags' are str.
value of 'nutrients' and 'portion' are list x.
factor of list x is dict

u'description': u'Cheese, caraway',
 u'group': u'Dairy and Egg Products',
 u'id': 1008,
 u'manufacturer': u'',
 u'nutrients': [{u'description': u'Protein',
   u'group': u'Composition',
   u'units': u'g',
   u'value': 25.18},
  {u'description': u'Total lipid (fat)',
   u'group': u'Composition',
   u'units': u'g',
   u'value': 29.2},
   ...
'''
db[0]['description'] #type is unicode

'''
输出：
u'Cheese, caraway'
'''

db[0]['nutrients'] #type is list

'''
输出：
[{u'description': u'Protein',
  u'group': u'Composition',
  u'units': u'g',
  u'value': 25.18},
 {u'description': u'Total lipid (fat)',
  u'group': u'Composition',
  u'units': u'g',
  u'value': 29.2},
  {..},{..},...{..}
]
'''

db[0]['nutrients'][0] #type is dict

'''
输出：
{u'description': u'Protein',
 u'group': u'Composition',
 u'units': u'g',
 u'value': 25.18}
'''

nutrients = DataFrame(db[0]['nutrients'])

info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)

#info1 = DataFrame(db[0], columns=info_keys)
info2 = DataFrame(db[0]['nutrients'], columns=info_keys)

nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

nutrients = nutrients.drop_duplicates() #去掉重复项

col_mapping = {'description' : 'food',
               'group'       : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)#对于info中的某些列名，重命名

col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False) #对于nutrients中的某些列名，重命名
# DataFrame1 = DF1.rename(columns = dict1, copy=False)

ndata = pd.merge(nutrients, info, on='id', how='outer') #合并nutrients和info两个DataFrame

result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
#result['Zinc, Zn'].order().plot(kind='barh', stacked = True)

#x = result['Zinc, Zn'].order()


by_nutrient = ndata.groupby(['nutgroup', 'nutrient']) #type is pandas.core.groupby.DataFrameGroupBy

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())

max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

# make the food a little smaller
max_foods.food = max_foods.food.str[:50] # type of max_foods is DataFrame, type of max_foods.food is Series

by_nutrient = ndata.groupby(['nutrient'])

max_foods.ix['Amino Acids']['food']