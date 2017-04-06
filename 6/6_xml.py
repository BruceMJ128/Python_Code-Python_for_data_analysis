# -*- coding: utf-8 -*-
from lxml import objectify
from pandas import DataFrame


path = 'data/ch06/Performance_XML_Data/Performance_MNR.xml'
parsed = objectify.parse(open(path)) #lxml.etree._ElementTree
root = parsed.getroot() #lxml.objectify.ObjectifiedElement

data = []

skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ','DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR: #root.INDICATOR返回一个用于产生<INDICATOR>XML元素的生成器, type of elt is lxml.objectify.ObjectifiedElement
    el_data = {}
    for child in elt.getchildren(): #type of child is lxml.objectify.StringElement
        if child.tag in skip_fields: #type of child.tag is str, 相当于数据的行名
            continue  #如continue，则直接跳到了下一个循环
        el_data[child.tag] = child.pyval #child.pyval是每行的数据，类型在本例中为str,float,unicode
    data.append(el_data)

#data的具体数据分层格式为 [{tag1: value1, tag2: value2,...},{},...]

perf = DataFrame(data)

