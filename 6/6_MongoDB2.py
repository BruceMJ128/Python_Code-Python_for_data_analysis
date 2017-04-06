# -*- coding: utf-8 -*-
import pymongo
import json
import requests

from pandas import DataFrame

tweets={}
tweets2={}

con = pymongo.Connection('localhost',port=27017) 

db_tweets = con.db.tweets
db_tweets2 = con.db.tweets2
dict2 = {'name': 'Refactor'}
cursor=db_tweets.find(dict2)
cursor2=db_tweets2.find(dict2) 

key2=['color','id', 'name','url']

result = DataFrame(list(cursor), columns=key2)
result2 = DataFrame(list(cursor2), columns=key2)

#result 和 result2不一样的原因，是因为在6_MongoDB.py中对于 db_tweets = con.db.tweets已经进行了初始化
#至于背后的原理，目前还不清楚，待学习