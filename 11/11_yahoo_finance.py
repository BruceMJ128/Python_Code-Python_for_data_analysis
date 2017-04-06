# -*- coding: utf-8 -*-
import pandas_datareader

import pandas_datareader.data as web
import datetime

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2013, 1, 27)

f = web.DataReader("F", 'yahoo', start, end)

f.ix['2010-01-04']


f1=web.get_data_yahoo('AAPL','1/1/2014','20/8/2015')

############################

amzn = web.get_quote_yahoo('AMZN')  #获得股票概况

amzn

'''
          PE change_pct   last  short_ratio    time
AMZN  182.34     -0.13%  795.9         1.59  4:00pm
'''

import pandas_datareader.data as web

import datetime

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2015, 5, 9)

f2 = web.DataReader('AAPL', 'yahoo-actions', start, end)  #查看股息 dividend 和股票分割stock split



import pandas_datareader.data as web

import datetime

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2013, 1, 27)
#f = web.DataReader("F", 'yahoo-dividends', start, end) #报错，data_source='yahoo-dividends' is not implemented 
f

 #Options.get_all_data()报错， get_all_data()不可用
''' 
from pandas_datareader.data import Options

aapl = Options('aapl', 'yahoo')

data = aapl.get_all_data()

data.iloc[0:5, 0:5]

data.loc[(100, slice(None), 'put'),:].iloc[0:5, 0:5]

data.loc[(100, slice(None), 'put'),'Vol'].head()

#If you don’t want to download all the data, more specific requests can be made.


import datetime

expiry = datetime.date(2016, 1, 1)

data = aapl.get_call_data(expiry=expiry)

data.iloc[0:5:, 0:5]

aapl.expiry_dates

data = aapl.get_call_data(expiry=aapl.expiry_dates[0])
data.iloc[0:5:, 0:5]

data = aapl.get_near_stock_price(expiry=aapl.expiry_dates[0:3])
data.iloc[0:5:, 0:5]

'''

# Google Finance
# 因为在中国地区对于google的限制，google.finance无效

'''


import pandas_datareader.data as web

import datetime

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2013, 1, 27)

#f = web.DataReader("F", 'google', start, end) 因为在中国地区对于google的限制，google.finance无效

#f.ix['2010-01-04']


import pandas_datareader.data as web

#q = web.get_quote_google(['AMZN', 'GOOG'])




from pandas_datareader.data import Options

goog = Options('goog', 'google')

data = goog.get_options_data(expiry=goog.expiry_dates[0])

data.iloc[0:5, 0:5]

'''

# Enigma #
import os

import pandas_datareader as pdr

#df = pdr.get_data_enigma('enigma.trade.ams.toxic.2015', os.getenv('ENIGMA_API_KEY'))
#pdr.get_data_enigma不能用

#df.columns


# FRED: Federal Reserve Bank of St. Louis 美国联邦储备银行

import pandas_datareader.data as web

import datetime

start = datetime.datetime(2010, 1, 1)

end = datetime.datetime(2013, 1, 27)

gdp = web.DataReader("GDP", "fred", start, end)

gdp.ix['2013-01-01']

inflation = web.DataReader(["CPIAUCSL", "CPILFESL"], "fred", start, end)

inflation.head()

# CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)
# CPILFESL: the core CPI (Consumer Price Index for All Urban Consumers: All Items Less Food & Energy [CPILFESL])


# Fama/French: Fama-French三因子模型，访问 Fama/French Data Library 数据库 get_available_datasets

from pandas_datareader.famafrench import get_available_datasets

import pandas_datareader.data as web

len(get_available_datasets())


ds = web.DataReader("5_Industry_Portfolios", "famafrench")

print(ds['DESCR'])

ds[4].ix['2016-07']

# World Bank

from pandas_datareader import wb

wb.search('gdp.*capita.*const').iloc[:,:2]  #compare the Gross Domestic Products per capita in constant dollars in North America

dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX'], start=2005, end=2008)
print(dat)
dat['NY.GDP.PCAP.KD'].groupby(level=0).mean()

wb.search('cell.*%').iloc[:,:2] #compare GDP to the share of people with cellphone contracts around the world.

ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']
dat = wb.download(indicator=ind, country='all', start=2011, end=2011).dropna()
dat.columns = ['gdp', 'cellphone']
print(dat.tail())

import numpy as np
import statsmodels.formula.api as smf  
#statsmodels package to assess the relationship between our two variables using ordinary least squares regression.

mod = smf.ols("cellphone ~ np.log(gdp)", dat).fit()
print(mod.summary())





# OECD: OECD Statistics are avaliable via DataReader
# 经济合作与发展组织（英语：Organization for Economic Co-operation and Developmen

import pandas_datareader.data as web

import datetime

df = web.DataReader('UN_DEN', 'oecd', end=datetime.datetime(2012, 1, 1)) #工会密度Trade Union Density” data which set code is “UN_DEN”.
df.columns

df[['Japan', 'United States']]



# Eurostat: Eurostat are avaliable via DataReader.

import pandas_datareader.data as web

df = web.DataReader("tran_sf_railac", 'eurostat')  #tran_sf_railac： Rail accidents by type of accident

df




'''

# 说明：
# import pandas_datareader.tsp 报错
# from pandas_datareader.oanda import 报错
# from pandas_datareader.nasdaq_trader 报错

######TSP#######

# 时间序列分析软件 TSP 是拥有超过两千种以上全球公认的标准经济估计方法



#TSP Fund Data: Download mutual fund index prices for the TSP.

import pandas_datareader.tsp as tsp

tspreader = tsp.TSPReader(start='2015-10-1', end='2015-12-31')

tspreader.read()





########  Oanda  #########

# Oanda currency historical rate  #的互联网货币交易公司,也是外币兑换成当地货币服务 和货币信息服务领域的全球领导商

#Download currency historical rate from Oanda.


from pandas_datareader.oanda import get_oanda_currency_historical_rates
start, end = "2016-01-01", "2016-06-01"
quote_currency = "USD"
base_currency = ["EUR", "GBP", "JPY"]
df_rates = get_oanda_currency_historical_rates(start, end,quote_currency=quote_currency,base_currency=base_currency)
print(df_rates)




# Nasdaq Trader Symbol Definitions

from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
print(symbols.ix['IBM'])
'''