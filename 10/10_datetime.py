# -*- coding: utf-8 -*-
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))

#%matplotlib inline

from datetime import datetime
now = datetime.now()
now

now.year, now.month, now.day

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta

delta.days

delta.seconds

from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12) #12会变为days

start - 2 * timedelta(12)

stamp = datetime(2011, 1, 3)

str(stamp)

stamp.strftime('%Y-%m-%d') 

value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d') #用datetime.strptime()的方法赋值

datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

from dateutil.parser import parse
parse('2011-01-03')

parse('Jan 31, 1997 10:45 PM')

parse('6/12/2011', dayfirst=True) #6指的是day，否则6指的是month

datestrs

pd.to_datetime(datestrs) 
# note: output changed (no '00:00:00' anymore)

idx = pd.to_datetime(datestrs + [None]) #type is pandas.tseries.index.DatetimeIndex
idx

idx[2] #pandas.tslib.NaTType

idx[1] #pandas.tslib.Timestamp

pd.isnull(idx)

#---------------------Time Series Basics--------------------------

from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
ts

type(ts)
# note: output changed to "pandas.core.series.Series"

ts.index

ts + ts[::2]

ts.index.dtype
# note: output changed from dtype('datetime64[ns]') to dtype('<M8[ns]')

stamp = ts.index[0] # type is pandas.tslib.Timestamp
stamp 
# note: output changed from <Timestamp: 2011-01-02 00:00:00> to Timestamp('2011-01-02 00:00:00')

#---------------------------Indexing, selection, subsetting---------------------------------

stamp = ts.index[2]
ts[stamp]

ts['1/10/2011']

ts['20110110']

longer_ts = Series(np.random.randn(1000),
index=pd.date_range('1/1/2000', periods=1000)) #起始点位1/1/2000，1000天
longer_ts

longer_ts['2001'] #显示整年

longer_ts['2001-05'] #显示整月

ts[datetime(2011, 1, 7):]

ts

ts['1/6/2011':'1/11/2011']

ts.truncate(after='1/9/2011') #截取1/9/2011日期之前的信息
ts.truncate(before='1/9/2011') #截取1/9/2011日期之后的信息
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED') #pandas.tseries.index.DatetimeIndex
long_df = DataFrame(np.random.randn(100, 4),
index=dates,
columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.ix['5-2001'] #截取2001年5月的信息

#-------------------------Time series with duplicate indices--------------------------------
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
'1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
dup_ts

dup_ts.index.is_unique

dup_ts['1/3/2000'] # not duplicated

dup_ts['1/2/2000'] # duplicated

grouped = dup_ts.groupby(level=0) #按照轴level=0，即行名
grouped.mean() 

grouped.count()

#--------------------Date ranges, Frequencies, and Shifting-------------------------------------
ts
ts.resample('D')

#-----------------------Generating date ranges----------------------------------

index = pd.date_range('4/1/2012', '6/1/2012')  #type is pandas.tseries.index.DatetimeIndex
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour


four_hours = Hour(4)
four_hours


pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')


Hour(2) + Minute(30)


pd.date_range('1/1/2000', periods=10, freq='1h30min')


#---------------------------Frequencies and Date Offsets------------------------------

from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour


four_hours = Hour(4)
four_hours


pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')


Hour(2) + Minute(30)


pd.date_range('1/1/2000', periods=10, freq='1h30min')


#### Week of month dates


rng = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI') #WOM: week of month, WOM-3FRI: 每个月第三个星期五
list(rng)

### Shifting (leading and lagging) data


ts = Series(np.random.randn(4),index=pd.date_range('1/1/2000', periods=4, freq='M')) #M: MonthEnd 每月最后一个日历日
ts


ts.shift(2) #索引不变，每个日期对应的数据向后移动一位


ts.shift(-2)

ts / ts.shift(1) - 1

ts1 = ts.shift(3, freq='D') #index向后移动三天

ts.shift(1, freq='3D') #结果与ts.shift(3, freq='D')一样

ts.shift(2, freq='M') #数值不变，索引日期，以MonthEnd的频率，向后移动两位

ts1.shift(2,freq='M') #结果与ts.shift(2, freq='M')一样，以MonthEnd的频率向后移动


ts.shift(1, freq='90T') #向后移动90分钟


#### Shifting dates with offsets


from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()


now + MonthEnd() #now最近的一个MonthEnd()


now + MonthEnd(2) #now时刻下个月的MonthEnd


offset = MonthEnd()
offset.rollforward(now) #now时刻向后推一个MonthEnd()


offset.rollback(now) #now时刻向前推一个MonthEnd()

now3 = datetime(2011,11,30,10,59,59)  #datetime(yyyy,mm,dd,min,ss)至少要出现到day

offset.rollforward(now3) # Timestamp('2011-11-30 10:59:59')，当now3出现分和秒的时候，MonthEnd().roolforward(now3)失效了，但now3+MonthEnd()仍有效

offset.rollback(now3)  # Timestamp('2011-11-30 10:59:59')

now3+MonthEnd() # Timestamp('2011-12-31 10:59:59')

now3+MonthEnd(2)  # Timestamp('2012-01-31 10:59:59')


ts = Series(np.random.randn(20), index=pd.date_range('1/15/2000', periods=20, freq='4d')) 
ts.groupby(offset.rollforward).mean() #自定义了offset = MonthEnd()

ts.groupby(MonthEnd().rollforward).mean()   #对于1月份的所有数据，加和求平均，将1月份的均值赋给MonthEnd, 1月31号

#type(offset.rollforward) is instancemethod

type(offset.rollforward(now)) #type is pandas.tslib.Timestamp

ts.resample('M', how='mean')  #等效于ts.groupby(MonthEnd().rollforward).mean() ，求每月均值

## Time Zone Handling


import pytz   #时区信息保存在 库 pytz中
pytz.common_timezones[-5:]  #['US/Eastern', 'US/Hawaii', 'US/Mountain', 'US/Pacific', 'UTC']


tz = pytz.timezone('US/Eastern')
tz

tz1 = pytz.timezone('US/Hawaii')

### Localization and Conversion


rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)


print(ts.index.tz)  #此时未赋值时区，.tz输出为None


pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')


ts_utc = ts.tz_localize('UTC')
ts_utc


ts_utc.index


ts_utc.tz_convert('US/Eastern')  #Series.tz_convert(' ')转换时区


ts_eastern = ts.tz_localize('US/Eastern') #Series.tz_localize(' ')定义时区, Series2 = Series1.tz_localize(' ') ，直接将Series1的时区转化

'''
ts_eastern
Out[6]: 
2012-03-09 09:30:00-05:00   -0.506629
2012-03-10 09:30:00-05:00   -0.673705
2012-03-11 09:30:00-04:00   -0.382820
2012-03-12 09:30:00-04:00    0.577853
2012-03-13 09:30:00-04:00    0.258054
2012-03-14 09:30:00-04:00   -0.084743
Freq: D, dtype: float64
'''


ts_eastern.tz_convert('UTC')  


ts_eastern2=ts_eastern.tz_convert('Europe/Berlin')  #type is pandas.core.series.Series 将Series的index转换为别的时区, Series2 = Series1.tz_convert(' ')

'''
ts_eastern2

2012-03-09 15:30:00+01:00    1.231444
2012-03-10 15:30:00+01:00   -0.339551
2012-03-11 14:30:00+01:00   -0.777978
2012-03-12 14:30:00+01:00    1.486255
2012-03-13 14:30:00+01:00   -1.203718
2012-03-14 14:30:00+01:00   -0.510967
'''

ts2= ts.index.tz_localize('Asia/Shanghai')  #type is pandas.tseries.index.DatetimeIndex，生成一个DatetimeIndex

'''
ts2
DatetimeIndex(['2012-03-09 09:30:00+08:00', '2012-03-10 09:30:00+08:00',
               '2012-03-11 09:30:00+08:00', '2012-03-12 09:30:00+08:00',
               '2012-03-13 09:30:00+08:00', '2012-03-14 09:30:00+08:00'],
              dtype='datetime64[ns, Asia/Shanghai]', freq='D')
'''



### Operations with time zone-aware Timestamp objects


stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
stamp_utc.tz_convert('US/Eastern')


stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow


stamp_utc.value  #value为时间戳，该值不会随着时区变化而改变，是从1970年1月1日起算起的纳秒数


stamp_utc.tz_convert('US/Eastern').value


# 30 minutes before DST transition
from pandas.tseries.offsets import Hour
stamp_0 = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
stamp_0

stamp_0 + Hour()


stamp_1 = pd.Timestamp('2012-03-11 01:30', tz='US/Eastern') #冬令时 切 夏令时
#Timestamp('2012-03-11 01:30:00-0500', tz='US/Eastern')

stamp_1 + Hour()
#Timestamp('2012-03-11 03:30:00-0400', tz='US/Eastern')

# 90 minutes before DST transition
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern') #夏令时 切 冬令时
stamp
# Timestamp('2012-11-04 00:30:00-0400', tz='US/Eastern')

stamp1 = stamp + Hour()  #注意stamp1和stamp2的区别
# Timestamp('2012-11-04 01:30:00-0400', tz='US/Eastern')

stamp2 = stamp + 2 * Hour() #夏令时更改为冬令时的时间变化
#Timestamp('2012-11-04 01:30:00-0500', tz='US/Eastern'),注意stamp往后延后了两个小时，但是时间只变了一个小时，时区往后延迟了一个


### Operations between different time zones


rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2 #二者相加，index取二者的并集，有交集的部分value相加，没有交集的部分，则为NaN

#ts+ts1 会 fail，因为ts没有指定时区，而ts1指定了时区，如果给ts指定'UTC'二者是可以加和的

result.index


## Periods and Period Arithmetic


p = pd.Period(2007, freq='A-DEC') #每年12月的最后一个日历日，12月31号
p


p + 5  #按年加5年，Period('2012', 'A-DEC')


p - 2


pd.Period('2014', freq='A-DEC') - p


rng = pd.period_range('1/1/2000', '6/30/2000', freq='M') #M 每个月最后一个日历日
rng


Series(np.random.randn(6), index=rng)


values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC') #Q-Dec指定月份的所在季度的最后一个月的最后一个日历日

#index = pd.PeriodIndex(list, freq = str)

index

### Period Frequency Conversion


p = pd.Period('2007', freq='A-DEC')
p.asfreq('M', how='start')


p.asfreq('M', how='end')


p = pd.Period('2007', freq='A-JUN')
p.asfreq('M', 'start')


p.asfreq('M', 'end')


p = pd.Period('Aug-2007', 'M')
p.asfreq('A-JUN')


rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


ts.asfreq('M', how='start')


ts.asfreq('B', how='end')

### Quarterly period frequencies


p = pd.Period('2012Q4', freq='Q-JAN') #Q-JAN，就决定了最后一天是2012.01.31，然后往前倒推
p
# Period('2012Q4', 'Q-JAN')

p.asfreq('D', 'start')
# Period('2011-11-01', 'D')

p.asfreq('D', 'end')
# Period('2012-01-31', 'D')

p1 = pd.Period('2012Q4')

p1
#Period('2012Q4', 'Q-DEC')

p1.asfreq('D', 'start')
#Period('2012-10-01', 'D')

p1.asfreq('D', 'end')
#Period('2012-12-31', 'D')


p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60 #B: 工作日，e：end ； T：分钟，s：start，16*60决定了下午四点
p4pm


p4pm.to_timestamp()


rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
ts

new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
ts

### Converting Timestamps to Periods (and back)


rng = pd.date_range('1/1/2000', periods=3, freq='M')
ts = Series(randn(3), index=rng)
pts = ts.to_period()
ts


pts


rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = Series(randn(6), index=rng)
ts2.to_period('M')

pts = ts.to_period()
pts

pts.to_timestamp(how='end')

### Creating a PeriodIndex from arrays


data = pd.read_csv('data/ch08/macrodata.csv')
data.year


data.quarter


index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
index


data.index = index
data.infl

## Resampling and Frequency Conversion


rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(randn(len(rng)), index=rng)
ts.resample('M', how='mean')

ts.resample('M', how='mean', kind='period')

### Downsampling


rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
ts


ts.resample('5min', how='sum')
# note: output changed (as the default changed from closed='right', label='right' to closed='left', label='left'


ts.resample('5min', how='sum', closed='left')


ts.resample('5min', how='sum', closed='left', label='left')  #closed说的是临界值5min属于左边区间还是右边区间
'''
2000-01-01 00:00:00    10
2000-01-01 00:05:00    35
2000-01-01 00:10:00    21
'''

ts.resample('5min', how='sum', closed='left', label='right')
'''
2000-01-01 00:05:00    10
2000-01-01 00:10:00    35
2000-01-01 00:15:00    21
'''

ts.resample('5min', how='sum', loffset='-1s')

#### Open-High-Low-Close (OHLC) resampling


ts.resample('5min', how='ohlc') #传入how='ohlc'得到四种数据
# note: output changed because of changed defaults

#### Resampling with GroupBy


rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
ts.groupby(lambda x: x.month).mean()


ts.groupby(lambda x: x.weekday).mean()

### Upsampling and interpolation


frame = DataFrame(np.random.randn(2, 4),
index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
columns=['Colorado', 'Texas', 'New York', 'Ohio'])  #指定周三开始算起
frame


df_daily = frame.resample('D')
df_daily


frame.resample('D', fill_method='ffill')


frame.resample('D', fill_method='ffill', limit=2)


frame.resample('W-THU', fill_method='ffill') #'W-THU'指定星期四开始算起

### Resampling with periods


frame = DataFrame(np.random.randn(24, 4),
index=pd.period_range('1-2000', '12-2001', freq='M'),
columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame

'''
	           Colorado	Texas	        New York	Ohio
2000/1/1	0.193671406	0.626337679	-0.66776433	0.124412335
2000/2/1	-0.150275261	0.018878225	0.482556561	0.565524357
2000/3/1	-0.195402698	-0.582896675	0.809607689	0.245660772
2000/4/1	0.215923423	-1.090665429	-0.255118719	-0.838814931
2000/5/1	0.836986699	-1.585078557	-1.251066078	0.12679132
2000/6/1	2.030312145	0.201645029	-0.466540637	0.339695028
2000/7/1	1.016130579	0.198928439	0.086031048	0.27543632
2000/8/1	-0.81152288	1.755511676	1.669119456	1.74905141
2000/9/1	0.041844624	-0.344045596	-1.023663993	-0.997747128
2000/10/1	1.242363005	-0.718569546	-2.021908834	0.38607389
2000/11/1	0.7623763	-0.730820645	-1.512104393	0.287517474
2000/12/1	1.105115837	1.079085314	-1.537279813	1.378101627
2001/1/1	-0.608868619	0.650289456	0.461925654	-0.229772609
2001/2/1	0.223135894	-0.655634837	-1.716685306	0.281586047
2001/3/1	0.312822654	-1.456444487	-1.192035601	-0.164302127
2001/4/1	2.41056683	0.215637577	-0.847642651	1.765443395
2001/5/1	0.143842909	0.821939795	-1.154491479	-1.175981359
2001/6/1	-0.349172692	0.212716743	1.431246767	1.283541503
2001/7/1	0.97058852	-0.617391958	-0.9058075	-0.637983336
2001/8/1	-2.390285376	-0.52965918	-1.953134369	-0.314370075
2001/9/1	-0.654803847	-0.710482583	-0.844526861	1.034958745
2001/10/1	0.865445074	0.441312744	-0.064374685	-0.973152576
2001/11/1	-0.771059052	-0.792999775	0.688152027	-0.600528761
2001/12/1	1.555820948	0.134584555	1.704082605	-1.275980452

'''


annual_frame = frame.resample('A-DEC', how='mean') #此时'A-DEC'起到的作用是一年的轮回截止到十二月份为止，计算一轮的mean
annual_frame

'''
      Colorado     Texas  New York      Ohio
2000  0.523960 -0.097641 -0.474011  0.303475  #计算从2000-01月到2000-12的mean
2001  0.142336 -0.190511 -0.366108 -0.083878  #计算从2001-01月到2001-12的mean
'''

annual_frame = frame.resample('A-Nov', how='mean')
annual_frame
'''
      Colorado     Texas  New York      Ohio
2000  0.471128 -0.204616 -0.377350  0.205782  #计算从2000-01月到2000-11的mean
2001  0.104777 -0.111803 -0.636221  0.137295  #计算从2000-12月到2001-11的mean
2002  1.555821  0.134585  1.704083 -1.275980  #计算从2001-12月到2002-11的mean
'''


# Q-DEC: Quarterly, year ending in December
annual_frame.resample('Q-DEC', fill_method='ffill')
# note: output changed, default value changed from convention='end' to convention='start' + 'start' changed to span-like
# also the following cells


annual_frame.resample('Q-DEC', fill_method='ffill', convention='start') #'Q-DEC' 以指定月份结束的年度

annual_frame.resample('Q-MAR', fill_method='ffill')

## Time series plotting


close_px_all = pd.read_csv('data/ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B', fill_method='ffill')
close_px.info()


close_px['AAPL'].plot()


close_px.ix['2009'].plot()


close_px['AAPL'].ix['01-2011':'03-2011'].plot()


appl_q = close_px['AAPL'].resample('Q-DEC', fill_method='ffill') 
#‘Q-DEC’决定了12-31肯定会出现在结果中，且frequency 为季度，其他月份按照freq推导
#但最后一个数字是错的，close_px中并没有2011/12/31号，但appl_q中有2011/12/31号，值为close_px的最后一天2011/10/14的值
#从2003-03-31 ~ 2011-12-31
appl_q.ix['2009':].plot()

appl_q1 = close_px['AAPL'].resample('Q-FEB', fill_method='ffill') #其实点为1月31号
#从2003-02-28 ~ 2011-11-30

appl_q2 = close_px['AAPL'].resample('Q-APR', fill_method='ffill')
#从2003-01-31，2003-04-30 ~ 2011-11-30



## Moving window functions


close_px = close_px.asfreq('B').fillna(method='ffill')


close_px.AAPL.plot()
pd.rolling_mean(close_px.AAPL, 250).plot()


plt.figure()


appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=10)
appl_std250[5:12]


appl_std250.plot()


# Define expanding mean in terms of rolling_mean
#expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)


pd.rolling_mean(close_px, 60).plot(logy=True)


plt.close('all')


### Exponentially-weighted functions

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,figsize=(12, 7))

aapl_px = close_px.AAPL['2005':'2009']

ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)
ewma60 = pd.ewma(aapl_px, span=60)

aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')

#可以看到60日移动平均ma60和60日指数加权平均ewma60，大致相似，具体数字上略有不同

### Binary moving window functions


close_px
spx_px = close_px_all['SPX']

spx_rets = spx_px / spx_px.shift(1) - 1 #用2003-01-03的值除以2003-01-02的值
returns = close_px.pct_change()  #等于spx_rets
corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr.plot()

corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr.plot()


### User-defined moving window functions


from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
result.plot()

## Performance and Memory Usage Notes


rng = pd.date_range('1/1/2000', periods=10000000, freq='10ms')
ts = Series(np.random.randn(len(rng)), index=rng)
ts

ts.resample('15min', how='ohlc') #每隔15分钟取个点，然后找这个15分钟内的open, high, low, close
ts.resample('15min', how='ohlc').info()


#%timeit ts.resample('15min', how='ohlc')


rng = pd.date_range('1/1/2000', periods=10000000, freq='1s')
ts = Series(np.random.randn(len(rng)), index=rng)
#%timeit ts.resample('15s', how='ohlc')
