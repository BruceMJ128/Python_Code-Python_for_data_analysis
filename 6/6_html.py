# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np; import pylab as pl
import urllib2
from pandas import Series

from lxml.html import parse
from urllib2 import urlopen

from pandas import ExcelFile

parsed = parse(urlopen('http://finance.yahoo.com/quote/AAPL/analysts?p=AAPL')) #parsed type is lxml.etree._ElementTree

doc = parsed.getroot() #doc type is lxml.html.HtmlElement

links = doc.findall('.//a') #links type is list, html中链接是a标签，使用findall('.//a')得到文档中的所有URL链接
tables = doc.findall('.//table')

       

'''
lnk = links[2]


lnk.get('href')  #list.get('href')是提取URL链接格式的方法, lnk.get('href') 数据类型为str
Out[38]: 'https://www.flickr.com/'

lnk.text_content()  #list.text_content()是针对显示文本的提取方法
Out[39]: 'Flickr'


urls = [lnk.get('href') for lnk in doc.findall('.//a')] #获取doc中的所有URL

tables = doc.findall('.//table') #获取doc中的表格, tables type is list，元素为HtmlElement
calls=tables[0]  #calls type is lxml.html.HtmlElement

rows = calls.findall('.//tr') #获取call中的tr(数据行)??, rows type is list, 元素为tr

def _unpack(row, kind='td'): #函数对象为tr数据行，获取数据行单元格内文件，标题行 kind='th';数据行 kind='td'
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts] #返回值为list类型

#_unpack(rows[0], kind='td')

from pandas.io.parsers import TextParser  #所有步骤合在一起，将数据转换为一个DataFrame

def parse_options_data(table):
    rows = table.findall('.//tr') #由tr组成的list
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk() #pandas中的TextParser类，将可以将字符串转化为数字的数据转化为浮点数格式，同样适用于read_csv等

call_data = parse_options_data(calls)
#put_data = parse_options_data(puts)
call_data[:10]
'''