# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np; import pylab as pl

from pandas import DataFrame

frame = pd.read_csv('data/ch06/ex1.csv') #frame of df is DataFrame

frame.to_pickle('data/ch06/frame_pickle') #将DataFrame保存为pickle格式

df = pd.read_pickle('data/ch06/frame_pickle') #type of df is DataFrame


store = pd.HDFStore('mydata.h5') 
#type of store is pandas.io.pytables.HDFStore, 并将store存为mydata.h5文件
# .h5文件有点类似于字典
store['obj1'] = frame # type of frame is pandas.core.frame.DataFrame
store['obj1_col'] = frame['a'] #type of frame['a'] is pandas.core.series.Series

xls_file = pd.ExcelFile('data/ch06/data.xlsx') 
#导入Excel文件，type of xls_file is pandas.io.excel.ExcelFile 
table = xls_file.parse('Sheet1') 
#存放在Sheet1工作表中的数据可以通过parse读取到DataFrame中

import requests
url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url) # type of resp is requests.models.Response

import json
data = json.loads(resp.text) #使用json将url的Response转化为list, type of data is list
#data的数据结构： [{key1:v1,key2:v2,...},{ ..}]

key = data[0].keys() 

tweets = DataFrame(data) #将data的从list格式转变为DataFrame，每个dict变为一行

import requests
import json
import os
import pymongo
import time

con = pymongo.Connection('localhost',port=27017)
#client =pymongo.MongoClient('localhost', 27017)

