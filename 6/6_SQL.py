# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np; import pylab as pl
import urllib2
from pandas import Series; from pandas import DataFrame

from lxml.html import parse
from urllib2 import urlopen

import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL,     d INTEGER
);"""
con = sqlite3.connect(':memory:') #创建数据库在内存中
con.execute(query) #执行sql语句
con.commit()  #事务提交

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahhassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?,?,?,?)"

con.executemany(stmt, data) #stmt是语法，data是具体数据
con.commit() #提交刚才的插入语句，之后才生效


cursor = con.cursor()
con.commit()

cursor.execute('select * from test')

rows = cursor.fetchall()

df = DataFrame(rows, columns = zip(*cursor.description)[0])



con.row_factory = sqlite3.Row
c = con.cursor()  #c 为cx的光标, sqlite3.Cursor
c.execute('select * from test')
r = c.fetchone()

import pandas.io.sql as sql

sql1 = sql.read_sql('select * from test', con)

#import pymongo

