# -*- coding: utf-8 -*-
import pymongo
import json
import requests

from pandas import DataFrame

tweets={}

con = pymongo.Connection('localhost',port=27017) 
#type of con is pymongo.connection.Connection，27017是mongodb默认端口
db_tweets = con.db.tweets 
#type of db_tweets is pymongo.collection.Collection，MongoDB的文档被组织在数据库的集合(collection)中，本例中，tweets作为DataFrame，其数据被保存进MongoDB数据库中，以db_tweets形式(connection)被调用和管理

url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url) # type of resp is requests.models.Response
data = json.loads(resp.text)

tweets = DataFrame(data)

for x in data:
    dict_x = dict(x)
    db_tweets.save(dict_x) #没运行一次，就往本地的mongodb数据库中写入一次文件，导致后来重复文件越来越多，而且在本程序结束后，信息不会被销毁，其他程序调用，依然保留
'''
'''
dict2 = {'color':'e10c02'}
cursor=db_tweets.find(dict2) #迭代器pymongo.cursor.Cursor,查找含有某一子字典的元素

key2=['color','id', 'name','url']

result = DataFrame(list(cursor), columns=key2)#查找cursor指向的元素，并保存key2指定的列

