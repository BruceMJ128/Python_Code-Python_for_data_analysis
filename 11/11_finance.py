# -*- coding: utf-8 -*-
# Financial and Economic Data Applications


from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 6))


#%matplotlib inline


#%pwd

## Data munging topics

### Time series and cross-section alignment


close_px = pd.read_csv('data/ch11/stock_px.csv', parse_dates=True, index_col=0)
volume = pd.read_csv('data/ch11/volume.csv', parse_dates=True, index_col=0)
prices = close_px.ix['2011-09-05':'2011-09-14', ['AAPL', 'JNJ', 'SPX', 'XOM']]
volume = volume.ix['2011-09-05':'2011-09-12', ['AAPL', 'JNJ', 'XOM']]


prices


volume


prices * volume


vwap = (prices * volume).sum() / volume.sum()


vwap


vwap.dropna()


prices.align(volume, join='inner') #结果为元组，第一个元素为prices的DF，第二元素为volume的DF


s1 = Series(range(3), index=['a', 'b', 'c'])
s2 = Series(range(4), index=['d', 'b', 'c', 'e'])
s3 = Series(range(3), index=['f', 'a', 'c'])
DataFrame({'one': s1, 'two': s2, 'three': s3})


DataFrame({'one': s1, 'two': s2, 'three': s3}, index=list('face')) #只显示字母f、a、c、e对应的元素

### Operations with time series of different frequencies
# 进行resample 和 reindex

ts1 = Series(np.random.randn(3),index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
ts1


ts1.resample('B')


ts1.resample('B', fill_method='ffill') #B means business day


dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18','2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)
ts2


ts1.reindex(ts2.index, method='ffill')


ts2 + ts1.reindex(ts2.index, method='ffill')


#### Using periods instead of timestamps


gdp = Series([1.78, 1.94, 2.08, 2.01, 2.15, 2.31, 2.46],index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.025, 0.045, 0.037, 0.04],index=pd.period_range('1982', periods=4, freq='A-DEC'))
gdp


infl


infl_q = infl.asfreq('Q-SEP', how='end')


infl_q


infl_q.reindex(gdp.index, method='ffill')

### Time of day and "as of" data selection


# Make an intraday date range and time series 生成一个交易日内的日期范围和时间序列
rng = pd.date_range('2012-06-01 09:30', '2012-06-01 15:59', freq='T')
# Make a 5-day series of 9:30-15:59 values
rng = rng.append([rng + pd.offsets.BDay(i) for i in range(1, 4)]) 
#包括2012-06-01,2012-06-04,2012-06-05,2012-06-06 四天

ts = Series(np.arange(len(rng), dtype=float), index=rng)
ts


from datetime import time
ts[time(10, 0)] #显示每天10:00:00


ts.at_time(time(10, 0))


ts.between_time(time(10, 0), time(10, 1))


np.random.seed(12346)


# Set most of the time series randomly to NA
indexer = np.sort(np.random.permutation(len(ts))[700:])
irr_ts = ts.copy()
irr_ts[indexer] = np.nan
irr_ts['2012-06-01 09:50':'2012-06-01 10:00']


selection = pd.date_range('2012-06-01 10:00', periods=4, freq='B')
irr_ts.asof(selection) #本来2012-06-01 10:00位置数值可能为NaN，.asof（）之后变为取之前最近一个数字

'''
2012/6/6 9:57	1197
2012/6/6 9:58	
2012/6/6 9:59	
2012/6/6 10:00	

irr_ts.asof(selection)

2012-06-06 10:00:00    1197.0
将2012/6/6 9:57的值赋给了 2012/6/6 10:00
'''

### Splicing together data sources


data1 = DataFrame(np.ones((6, 3), dtype=float),columns=['a', 'b', 'c'],index=pd.date_range('6/12/2012', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2,columns=['a', 'b', 'c'],index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
spliced


data2 = DataFrame(np.ones((6, 4), dtype=float) * 2,columns=['a', 'b', 'c', 'd'],index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
spliced


spliced_filled = spliced.combine_first(data2)
spliced_filled


spliced.update(data2, overwrite=False)


spliced


cp_spliced = spliced.copy()
cp_spliced[['a', 'c']] = data1[['a', 'c']]
cp_spliced


### Return indexes and cumulative returns


import pandas_datareader.data as web



price = web.get_data_yahoo('AAPL', '2011-01-01')['Adj Close'] #Adj Close: 已调整收盘价 adjusted closing price
price[-5:]


price['2011-10-03'] / price['2011-3-01'] - 1


returns = price.pct_change()  #DataFrame.pct_change() 相等于 DF / DF.shift(1) - 1
ret_index = (1 + returns).cumprod() #.cumprod() 简单计算综合考虑派息、稀释股票后的收益指数
ret_index[0] = 1 # Set first value to 1
ret_index


m_returns = ret_index.resample('BM', how='last').pct_change()
m_returns['2012']


m_rets = (1 + returns).resample('M', how='prod', kind='period') - 1
m_rets['2012']

#returns[dividend_dates] += dividend_pcts #dividend_dates为股息派发日，如果知道的话，则计算总股息收益


## Group transforms and analysis


pd.options.display.max_rows = 100
pd.options.display.max_columns = 10
np.random.seed(12345)


import random; random.seed(0)
import string

N = 1000
def rands(n):  #随机产生5位的股票代码
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in xrange(n)])
tickers = np.array([rands(5) for _ in xrange(N)])


M = 500
df = DataFrame({'Momentum' : np.random.randn(M) / 200 + 0.03,
'Value' : np.random.randn(M) / 200 + 0.08,
'ShortInterest' : np.random.randn(M) / 200 - 0.02},
index=tickers[:M])


ind_names = np.array(['FINANCIAL', 'TECH'])
sampler = np.random.randint(0, len(ind_names), N)  #在0和len(ind_names)任选一个数，选N次，组成数组
industries = Series(ind_names[sampler], index=tickers,name='industry') #将股票分为两类


by_industry = df.groupby(industries) #按照industries中的分类，进行groupby
by_industry.mean()


by_industry.describe()


# Within-Industry Standardize
def zscore(group):
    return (group - group.mean()) / group.std()

df_stand = by_industry.apply(zscore)


df_stand.groupby(industries).agg(['mean', 'std'])


# Within-industry rank descending
ind_rank = by_industry.rank(ascending=False)  #获得排名的次序
ind_rank.groupby(industries).agg(['min', 'max'])


# Industry rank and standardize
by_industry.apply(lambda x: zscore(x.rank()))


### Group factor exposures


from numpy.random import rand
fac1, fac2, fac3 = np.random.rand(3, 1000) #fac1 为array，元素数量为1000

ticker_subset = tickers.take(np.random.permutation(N)[:1000])

# Weighted sum of factors plus noise
port = Series(0.7 * fac1 - 1.2 * fac2 + 0.3 * fac3 + rand(1000),index=ticker_subset)
factors = DataFrame({'f1': fac1, 'f2': fac2, 'f3': fac3},index=ticker_subset)


factors.corrwith(port) #求相关性


pd.ols(y=port, x=factors).beta #最小二乘回归，暴露计算因子的权重


def beta_exposure(chunk, factors=None):
    return pd.ols(y=chunk, x=factors).beta


by_ind = port.groupby(industries)
exposures = by_ind.apply(beta_exposure, factors=factors)
exposures.unstack()

### Decile and quartile analysis


import pandas_datareader.data as web
data = web.get_data_yahoo('SPY', '2006-01-01')
data.info()


px = data['Adj Close']
returns = px.pct_change() #pct-change()看与之前的变化比例

def to_index(rets): #type of rets is Series
    index = (1 + rets).cumprod()  #.cimprod 累计求积
    first_loc = max(index.index.get_loc(index.idxmax()) - 1, 0) #index.index.get_loc(index.idxmax()) 获得最大值的位置
    index.values[first_loc] = 1  #将最大的值变为1，在图像中最高点之处出现一个值为1的点，即直线掉落再回升
    return index

def trend_signal(rets, lookback, lag):
    signal = pd.rolling_sum(rets, lookback, min_periods=lookback - 5)  #rolling_sum 即 moving sum
    return signal.shift(lag) #向后移动lag个单位


signal = trend_signal(returns, 100, 3)
trade_friday = signal.resample('W-FRI').resample('B', fill_method='ffill')
trade_rets = trade_friday.shift(1) * returns
trade_rets = trade_rets[:len(returns)]

plt.figure()
to_index(trade_rets).plot()


vol = pd.rolling_std(returns, 250, min_periods=200) * np.sqrt(250)

def sharpe(rets, ann=250):
    return rets.mean() / rets.std() * np.sqrt(ann)


cats = pd.qcut(vol, 4) #pd.qcut(Series, number)按照值分为number个区间
print('cats: %d, trade_rets: %d, vol: %d' % (len(cats), len(trade_rets), len(vol)))


trade_rets.groupby(cats).agg(sharpe)


## More example applications

### Signal frontier analysis


names = ['AAPL', 'GOOG', 'MSFT', 'DELL', 'GS', 'MS', 'BAC', 'C']
def get_px(stock, start, end):
    return web.get_data_yahoo(stock, start, end)['Adj Close']
px = DataFrame({n: get_px(n, None, None) for n in names})


#px = pd.read_csv('data/ch11/stock_px.csv')  #原始数据格式有问题，会使数据失效，试验get_px格式，从网上获取数据


plt.close('all')


px = px.asfreq('B').fillna(method='pad') # pad -> Fill values forward
rets = px.pct_change()
((1 + rets).cumprod() - 1).plot()


def calc_mom(price, lookback, lag):
    mom_ret = price.shift(lag).pct_change(lookback)
    ranks = mom_ret.rank(axis=1, ascending=False)
    demeaned = ranks.subtract(ranks.mean(axis=1), axis=0)
    return demeaned.divide(demeaned.std(axis=1), axis=0)


compound = lambda x : (1 + x).prod() - 1
daily_sr = lambda x: x.mean() / x.std()

def strat_sr(prices, lb, hold):
    # Compute portfolio weights
    freq = '%dB' % hold
    port = calc_mom(prices, lb, lag=1)

    daily_rets = prices.pct_change()

    # Compute portfolio returns
    port = port.shift(1).resample(freq, how='first')
    returns = daily_rets.resample(freq, how=compound)
    port_rets = (port * returns).sum(axis=1)
    
    return daily_sr(port_rets) * np.sqrt(252 / hold)


strat_sr(px, 70, 30)


from collections import defaultdict

lookbacks = range(20, 90, 5)
holdings = range(20, 90, 5)
dd = defaultdict(dict)
for lb in lookbacks:
    for hold in holdings:
        dd[lb][hold] = strat_sr(px, lb, hold)

ddf = DataFrame(dd)
ddf.index.name = 'Holding Period'
ddf.columns.name = 'Lookback Period'


import matplotlib.pyplot as plt

def heatmap(df, cmap=plt.cm.gray_r):  #画出热图 heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(df.values, cmap=cmap, interpolation='nearest')
    ax.set_xlabel(df.columns.name)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(list(df.columns))
    ax.set_ylabel(df.index.name)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(list(df.index))
    plt.colorbar(axim)


heatmap(ddf)


### Future contract rolling


pd.options.display.max_rows = 10


import pandas_datareader.data as web

# Approximate price of S&P 500 index
px = web.get_data_yahoo('SPY')['Adj Close'] * 10
px


from datetime import datetime
expiry = {'ESU2': datetime(2012, 9, 21),
'ESZ2': datetime(2012, 12, 21)}
expiry = Series(expiry).order()


expiry


np.random.seed(12347)
N = 200
walk = (np.random.randint(0, 200, size=N) - 100) * 0.25
perturb = (np.random.randint(0, 20, size=N) - 10) * 0.25
walk = walk.cumsum()

rng = pd.date_range(px.index[0], periods=len(px) + N, freq='B')
near = np.concatenate([px.values, px.values[-1] + walk])
far = np.concatenate([px.values, px.values[-1] + walk + perturb])
prices = DataFrame({'ESU2': near, 'ESZ2': far}, index=rng)


prices.tail()


def get_roll_weights(start, expiry, items, roll_periods=5):
# start : first date to compute weighting DataFrame
# expiry : Series of ticker -> expiration dates
# items : sequence of contract names

    dates = pd.date_range(start, expiry[-1], freq='B')
    weights = DataFrame(np.zeros((len(dates), len(items))),
    index=dates, columns=items)
    
    prev_date = weights.index[0]
    for i, (item, ex_date) in enumerate(expiry.iteritems()):
        if i < len(expiry) - 1:
            weights.ix[prev_date:ex_date - pd.offsets.BDay(), item] = 1
            roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
            periods=roll_periods + 1, freq='B')
            
            decay_weights = np.linspace(0, 1, roll_periods + 1)
            weights.ix[roll_rng, item] = 1 - decay_weights
            weights.ix[roll_rng, expiry.index[i + 1]] = decay_weights
        else:
            weights.ix[prev_date:, item] = 1
        
        prev_date = ex_date
    
    return weights


weights = get_roll_weights('6/1/2012', expiry, prices.columns)
weights.ix['2012-09-12':'2012-09-21']


rolled_returns = (prices.pct_change() * weights).sum(1)

### Rolling correlation and linear regression


aapl = web.get_data_yahoo('AAPL', '2000-01-01')['Adj Close']
msft = web.get_data_yahoo('MSFT', '2000-01-01')['Adj Close']

aapl_rets = aapl.pct_change()
msft_rets = msft.pct_change()


plt.figure()


pd.rolling_corr(aapl_rets, msft_rets, 250).plot()


plt.figure()


model = pd.ols(y=aapl_rets, x={'MSFT': msft_rets}, window=250)
model.beta


model.beta['MSFT'].plot()

