import pandas as pd; import numpy as np; import pylab as pl
import urllib2
from pandas import Series

from lxml.html import parse
from urllib2 import urlopen

from pandas import ExcelFile
from lxml.html import parse

parsed = parse(urlopen('http://finance.yahoo.com/quote/AAPL/options?p=AAPL'))
doc = parsed.getroot()
links = doc.findall('.//a')

lnk = links[28]

urls = [lnk.get('href') for lnk in doc.findall('.//a')]

tables = doc.findall('.//table')
calls = tables[0]
#puts = tables[1]

rows = calls.findall('.//tr')


def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

from pandas.io.parsers import TextParser

def parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()

call_data = parse_options_data(calls)
##put_data = parse_options_data(puts)
call_data[:10]