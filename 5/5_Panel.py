import pandas.io.data as web
import pandas as pd; import numpy as np; 

pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk, '1/1/2009', '6/1/2012')) for stk in ['AAPL','GOOG','MSFT','DELL']))

pdata2 = pdata.swapaxes('items', 'minor')
pdata3 = pdata2['Adj Close']

pdata4 = pdata2.ix[:, '6/1/2012',:]
pdata5 = pdata2.ix['Adj Close', '5/22/2012':, :]

stacked = pdata2.ix[:, '5/30/2012':, :].to_frame()

pdata6 = stacked.to_panel()

stacked2 = pdata2.to_frame()
stacked3 = pdata.to_frame()
