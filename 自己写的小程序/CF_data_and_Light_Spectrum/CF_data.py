# -*- coding: utf-8 -*-
import os
import re

from pandas import DataFrame, Series
import pandas as pd; import numpy as np
import xlrd

fo = open('C:/CF_data/input_path.txt')
path=fo.readline()

l = os.listdir(path) #获得文件夹中的文件名

pattern = r'([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)\(([0-9 ]+)&([A-Z0-9]+)\)([A-Z0-9- .]+)\.'
regex = re.compile(pattern, flags=re.IGNORECASE) #正则表达式的使用工具regex

index_line = ['NA1','Year','Month','Day','Lamp No.','10NC','Batch No.','Hrs','NA2','UVC','Power','Voltage','Current']

df=pd.DataFrame(columns=index_line)

for line in l:    
    fname = str(path+'/'+line) 
    bk = xlrd.open_workbook(fname)
    try:
        sh = bk.sheet_by_name("GUI")
    except:
         print "no sheet in %s named GUI" % fname
    UVC = sh.cell_value(17,12)
    P = sh.cell_value(17,13)
    V = sh.cell_value(17,14)
    I = sh.cell_value(17,15)
    data = [UVC,P,V,I]
    m_line = regex.split(line)    #将函数名str，变为包含多个信息的list
    m_line.extend(data)
    if len(m_line) == 13:
        S_temp = pd.Series(m_line, index = index_line)
        df = df.append(S_temp, ignore_index=True)
    else:
        error_line = {'NA1': m_line[0],'UVC':m_line[1],'Power':m_line[2],'Voltage':m_line[3],'Current':m_line[4]}
        df = df.append(error_line, ignore_index=True)
        continue
    
df.to_csv('C:/CF_data/CF_data.csv')