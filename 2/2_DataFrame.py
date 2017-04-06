from pandas import Series, DataFrame
import pandas as pd

data = {'name':['Carl', 'Peter', 'Lucy', 'Job'], 'age':[30,34,20,35], 'gender':['m','m','f','m']}
frame = DataFrame(data)

frame2 = DataFrame(data, columns = ['name', 'weight', 'gender', 'age'])

temp_weight  = frame2['weight']
temp_weight[frame2['name']=='Lucy'] = 50
