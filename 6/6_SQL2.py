# -*- coding: utf-8 -*-
import sqlite3

#cx = sqlite3.connect("C:\Users\310118430\OneDrive\Python\program\test.db")
cx = sqlite3.connect(":memory:")

cu=cx.cursor() #cx是数据库的connection(数据类型sqlite3.Connection), cu是数据库的游标（数据类型sqlite3.Cursor）

cu.execute("create table catalog (id integer primary key, pid iteger, name varchar(10) UNIQUE, nickname text NULL)")
cx.commit()

for t in [(0,10,'abc','Yu'),(1,20,'cba','Xu')]:
    cx.execute("insert into catalog values (?,?,?,?)", t)
cx.commit()

cu.execute("select * from catalog") #相当于将光标移动到数据库的初始位置

cu.fetchall()  #从光标位置开始向后查询，并输出

cu.execute("update catalog set name='Boy' where id=0") #修改内容

cu.execute("delete from catalog where id =1") #删除内容

cx.execute("insert into catalog values (?,?,?,?)", (2,30,'Jiao','Miao'))

x = u'于'
cu.execute("update catalog set name=? where id=0", x) #使用中文

cu.execute("select * from catalog")
for item in cu.fetchall():
    for element in item:
        print element,
    print

cx.row_factory = sqlite3.Row
c = cx.cursor()  #c 为cx的光标, sqlite3.Cursor
c.execute('select * from catalog')
r = c.fetchone()  #r type is sqlite3.Row 用法有点像list


