import pandas as pd; import numpy as np; import pylab as pl
import urllib2
from pandas import Series

from lxml.html import parse
from urllib2 import urlopen

frame = pd.read_csv('data/ch06/ex1.csv')

'''
store = pd.HDFStore('mydata.h5')
store['obj1']=frame
store['obj1_col']=frame['a']
'''