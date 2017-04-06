# -*- coding: utf-8 -*-
import pandas as pd;
import re

'''
val = 'a,b,  guido'

val1 = val.split(',')
pieces = [x.strip() for x in val.split(',')]


text = "foo    bar\t baz  \tqux"
re.split('\s+', text) # \s+ 描述一个或多个空白符

regex=re.compile('\s+') #type of regex is _sre.SRE_Pattern
regex = re.compile('\s+')
regex.split(text)

regex.findall(text)

'''


text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)

m = regex.search(text) 

print regex.match(text) #会报错，因为match只适用于纯字符串，不包含空格、空行等

pattern2 = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex2 = re.compile(pattern2, flags=re.IGNORECASE)

m2 = regex2.match('wesm@bright.net') #type of m2 is _sre.SRE_Match, match 适应于字符串格式，返回一个元组
m2_1 = regex2.match(text)#match不适用，会报错

m2.groups() #m.groups() 返回所有括号匹配的字符，以tuple格式。

regex2.findall('wesm@bright.net')#返回一个list，只包含一个元素，元素为元组
regex2.findall(text)#返回一个list，只包含4个元素，元素为元组

print(regex2.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
# .sub() 将匹配到的模式替换为指定字符串
'''
#输出为
Dave Username: dave, Domain: google, Suffix: com
Steve Username: steve, Domain: gmail, Suffix: com
Rob Username: rob, Domain: gmail, Suffix: com
Ryan Username: ryan, Domain: yahoo, Suffix: com
'''

regex3 = re.compile(r"""
    (?P<username>[A-Z0-9._%+-]+)
    @
    (?P<domain>[A-Z0-9.-]+)
    \.
    (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)

m3 = regex3.match('wesm@bright.net')

m3.groupdict()
#输出为：{'domain': 'bright', 'suffix': 'net', 'username': 'wesm'}

m3.groups()
#Out[50]: ('wesm', 'bright', 'net')