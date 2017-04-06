import pandas as pd; import numpy as np; import pylab as pl

from pandas import Series
from pandas import DataFrame

data = pd.read_csv('data/ch06/ex5.csv')
data.to_csv('data/ch06/out.csv')

dates = pd.date_range('1/1/2000', periods=7)
ts=Series(np.arange(7), index=dates)

import csv

f=open('data/ch06/ex7.csv')
reader = csv.reader(f)

lines = list(csv.reader(open('data/ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'

#reader = csv.reader(f, delimiter = '|')
reader = csv.reader(f, dialect = 1)

with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect = 1,delimiter = ';') #my_dialect
    writer.writerow(('one','two','three'))
    writer.writerow(('1','2','3'))
    writer.writerow(('4','5','6'))
    writer.writerow(('7','8','9'))

obj="""{"name": "Wes", "places_lived":["United States","Spain","Germany"], "pet": null, "siblings":[{"name":"Scott","age":25,"pet":"Zuko"},{"name":"Katie","age":33,"pet":"Cisco"}]}"""

import json

result = json.loads(obj)

asjson = json.dumps(result)

siblings = DataFrame(result['siblings'], columns=['name', 'age'])