# -*- coding: utf-8 -*-
<div id='noscript'> Jupyter Notebook requires JavaScript.<br> Please enable it to proceed. </div> 
ch10 Last Checkpoint: Last Wednesday at 3:16 PM (autosaved) 
Menu 
Python 2 
File
New Notebook
Python 2
Open...

Make a Copy...
Rename...
Save and Checkpoint

Revert to Checkpoint
Wednesday, January 4, 2017 3:16 PM

Print Preview
Download as
Notebook (.ipynb)
Python (.py)
HTML (.html)
Markdown (.md)
reST (.rst)
PDF via LaTeX (.pdf)

Trusted Notebook

Close and Halt
Edit
Cut Cells
Copy Cells
Paste Cells Above
Paste Cells Below
Paste Cells & Replace
Delete Cells
Undo Delete Cells

Split Cell
Merge Cell Above
Merge Cell Below

Move Cell Up
Move Cell Down

Edit Notebook Metadata

Find and Replace 
View
Toggle Header
Toggle Toolbar
Cell Toolbar
None
Edit Metadata
Raw Cell Format
Slideshow
Exercise
Insert
Insert Cell Above
Insert Cell Below
Cell
Run Cells
Run Cells and Select Below
Run Cells and Insert Below
Run All
Run All Above
Run All Below

Cell Type
Code
Markdown
Raw NBConvert

Current Outputs
Toggle
Toggle Scrolling
Clear
All Output
Toggle
Toggle Scrolling
Clear
Kernel
Interrupt
Restart
Restart & Clear Output
Restart & Run All
Reconnect

Change kernel
Python 2
Help
User Interface Tour
Keyboard Shortcuts

Notebook Help 
Markdown 

Python
IPython
NumPy
SciPy
Matplotlib
SymPy
pandas

About
CellToolbar
# Time series
Time series¶

x
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))


%matplotlib inline

## Date and Time Data Types and Tools
Date and Time Data Types and Tools¶

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
start + timedelta(12)


start - 2 * timedelta(12)

### Converting between string and datetime
Converting between string and datetime¶

x
stamp = datetime(2011, 1, 3)


x
str(stamp)


x
stamp.strftime('%Y-%m-%d')


x
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')


x
datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]


x
from dateutil.parser import parse
parse('2011-01-03')


x
parse('Jan 31, 1997 10:45 PM')


x
parse('6/12/2011', dayfirst=True)


x
datestrs


x
pd.to_datetime(datestrs)
# note: output changed (no '00:00:00' anymore)


x
idx = pd.to_datetime(datestrs + [None])
idx


x
idx[2]


pd.isnull(idx)

## Time Series Basics
Time Series Basics¶

x
from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
ts


x
type(ts)
# note: output changed to "pandas.core.series.Series"


x
ts.index


x
ts + ts[::2]


x
ts.index.dtype
# note: output changed from dtype('datetime64[ns]') to dtype('<M8[ns]')


x
stamp = ts.index[0]
stamp
# note: output changed from <Timestamp: 2011-01-02 00:00:00> to Timestamp('2011-01-02 00:00:00')

### Indexing, selection, subsetting
Indexing, selection, subsetting¶

x
stamp = ts.index[2]
ts[stamp]


x
ts['1/10/2011']


x
ts['20110110']


x
longer_ts = Series(np.random.randn(1000),
index=pd.date_range('1/1/2000', periods=1000))
longer_ts


x
longer_ts['2001']


longer_ts['2001-05']


x
ts[datetime(2011, 1, 7):]


ts


x
ts['1/6/2011':'1/11/2011']


x
ts.truncate(after='1/9/2011')


x
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
index=dates,
columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.ix['5-2001']

### Time series with duplicate indices
Time series with duplicate indices¶

x
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
'1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
dup_ts


x
dup_ts.index.is_unique


x
dup_ts['1/3/2000'] # not duplicated


x
dup_ts['1/2/2000'] # duplicated


x
grouped = dup_ts.groupby(level=0)
grouped.mean()


x
grouped.count()

## Date ranges, Frequencies, and Shifting
Date ranges, Frequencies, and Shifting¶

x
ts


x
ts.resample('D')

### Generating date ranges
Generating date ranges¶

x
index = pd.date_range('4/1/2012', '6/1/2012')
index


x
pd.date_range(start='4/1/2012', periods=20)


pd.date_range(end='6/1/2012', periods=20)


pd.date_range('1/1/2000', '12/1/2000', freq='BM')


pd.date_range('5/2/2012 12:56:31', periods=5)


pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)

### Frequencies and Date Offsets
Frequencies and Date Offsets¶

from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour


four_hours = Hour(4)
four_hours


pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')


Hour(2) + Minute(30)


pd.date_range('1/1/2000', periods=10, freq='1h30min')

#### Week of month dates
Week of month dates¶

rng = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI')
list(rng)

### Shifting (leading and lagging) data
Shifting (leading and lagging) data¶

ts = Series(np.random.randn(4),
index=pd.date_range('1/1/2000', periods=4, freq='M'))
ts


ts.shift(2)


ts.shift(-2)

ts / ts.shift(1) - 1

ts.shift(2, freq='M')


ts.shift(3, freq='D')


ts.shift(1, freq='3D')


ts.shift(1, freq='90T')

#### Shifting dates with offsets
Shifting dates with offsets¶

from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()


now + MonthEnd()


now + MonthEnd(2)


offset = MonthEnd()
offset.rollforward(now)


offset.rollback(now)


ts = Series(np.random.randn(20),
index=pd.date_range('1/15/2000', periods=20, freq='4d'))
ts.groupby(offset.rollforward).mean()


ts.resample('M', how='mean')

## Time Zone Handling
Time Zone Handling¶

import pytz
pytz.common_timezones[-5:]


tz = pytz.timezone('US/Eastern')
tz

### Localization and Conversion
Localization and Conversion¶

rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)


print(ts.index.tz)


pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')


ts_utc = ts.tz_localize('UTC')
ts_utc


ts_utc.index


ts_utc.tz_convert('US/Eastern')


ts_eastern = ts.tz_localize('US/Eastern')
ts_eastern.tz_convert('UTC')


ts_eastern.tz_convert('Europe/Berlin')


ts.index.tz_localize('Asia/Shanghai')

### Operations with time zone-aware Timestamp objects
Operations with time zone-aware Timestamp objects¶

stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
stamp_utc.tz_convert('US/Eastern')


stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow


stamp_utc.value


stamp_utc.tz_convert('US/Eastern').value


# 30 minutes before DST transition
from pandas.tseries.offsets import Hour
stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
stamp


stamp + Hour()


# 90 minutes before DST transition
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
stamp


stamp + 2 * Hour()

### Operations between different time zones
Operations between different time zones¶

rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
result.index

## Periods and Period Arithmetic
Periods and Period Arithmetic¶

p = pd.Period(2007, freq='A-DEC')
p


p + 5


p - 2


pd.Period('2014', freq='A-DEC') - p


rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
rng


Series(np.random.randn(6), index=rng)


values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
index

### Period Frequency Conversion
Period Frequency Conversion¶

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
Quarterly period frequencies¶

p = pd.Period('2012Q4', freq='Q-JAN')
p


p.asfreq('D', 'start')


p.asfreq('D', 'end')


p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
p4pm


p4pm.to_timestamp()


rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
ts


new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
ts

### Converting Timestamps to Periods (and back)
Converting Timestamps to Periods (and back)¶

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
Creating a PeriodIndex from arrays¶

data = pd.read_csv('ch08/macrodata.csv')
data.year


data.quarter


index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
index


data.index = index
data.infl

## Resampling and Frequency Conversion
Resampling and Frequency Conversion¶

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(randn(len(rng)), index=rng)
ts.resample('M', how='mean')


ts.resample('M', how='mean', kind='period')

### Downsampling
Downsampling¶

rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
ts


ts.resample('5min', how='sum')
# note: output changed (as the default changed from closed='right', label='right' to closed='left', label='left'


ts.resample('5min', how='sum', closed='left')


ts.resample('5min', how='sum', closed='left', label='left')


ts.resample('5min', how='sum', loffset='-1s')

#### Open-High-Low-Close (OHLC) resampling
Open-High-Low-Close (OHLC) resampling¶

ts.resample('5min', how='ohlc')
# note: output changed because of changed defaults

#### Resampling with GroupBy
Resampling with GroupBy¶

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
ts.groupby(lambda x: x.month).mean()


ts.groupby(lambda x: x.weekday).mean()

### Upsampling and interpolation
Upsampling and interpolation¶

frame = DataFrame(np.random.randn(2, 4),
index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame


df_daily = frame.resample('D')
df_daily


frame.resample('D', fill_method='ffill')


frame.resample('D', fill_method='ffill', limit=2)


frame.resample('W-THU', fill_method='ffill')

### Resampling with periods
Resampling with periods¶

frame = DataFrame(np.random.randn(24, 4),
index=pd.period_range('1-2000', '12-2001', freq='M'),
columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame[:5]


annual_frame = frame.resample('A-DEC', how='mean')
annual_frame


# Q-DEC: Quarterly, year ending in December
annual_frame.resample('Q-DEC', fill_method='ffill')
# note: output changed, default value changed from convention='end' to convention='start' + 'start' changed to span-like
# also the following cells


annual_frame.resample('Q-DEC', fill_method='ffill', convention='start')


annual_frame.resample('Q-MAR', fill_method='ffill')

## Time series plotting
Time series plotting¶

close_px_all = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B', fill_method='ffill')
close_px.info()


close_px['AAPL'].plot()


close_px.ix['2009'].plot()


close_px['AAPL'].ix['01-2011':'03-2011'].plot()


appl_q = close_px['AAPL'].resample('Q-DEC', fill_method='ffill')
appl_q.ix['2009':].plot()

## Moving window functions
Moving window functions¶

close_px = close_px.asfreq('B').fillna(method='ffill')


close_px.AAPL.plot()
pd.rolling_mean(close_px.AAPL, 250).plot()


plt.figure()


appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=10)
appl_std250[5:12]


appl_std250.plot()


# Define expanding mean in terms of rolling_mean
expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)


pd.rolling_mean(close_px, 60).plot(logy=True)


plt.close('all')

### Exponentially-weighted functions
Exponentially-weighted functions¶

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
figsize=(12, 7))

aapl_px = close_px.AAPL['2005':'2009']

ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)
ewma60 = pd.ewma(aapl_px, span=60)

aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')

### Binary moving window functions
Binary moving window functions¶

close_px
spx_px = close_px_all['SPX']


spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr.plot()


corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr.plot()

### User-defined moving window functions
User-defined moving window functions¶

from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
result.plot()

## Performance and Memory Usage Notes
Performance and Memory Usage Notes¶

rng = pd.date_range('1/1/2000', periods=10000000, freq='10ms')
ts = Series(np.random.randn(len(rng)), index=rng)
ts


ts.resample('15min', how='ohlc').info()


%timeit ts.resample('15min', how='ohlc')


rng = pd.date_range('1/1/2000', periods=10000000, freq='1s')
ts = Series(np.random.randn(len(rng)), index=rng)
%timeit ts.resample('15s', how='ohlc')

CloseExpandOpen in PagerClose


