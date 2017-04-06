import json
path = 'data/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]  

from pandas import DataFrame, Series
import pandas as pd; import numpy as np; import pylab as pl

frame = DataFrame(records)

tz_counts = frame['tz'].value_counts()

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz==''] = 'Unknown'

tz_counts2= clean_tz.value_counts()

results = Series([x.split()[0] for x in frame.a.dropna()])

cframe = frame[frame.a.notnull()]

operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')

by_tz_os = cframe.groupby(['tz',operating_system])

agg_counts = by_tz_os.size().unstack().fillna(0)

indexer = agg_counts.sum(1).argsort()

count_subset1 = agg_counts.take(indexer)[-10:]

# count_subset1.plot(kind = 'barh', stacked = True)

normed_subset = count_subset1.div(count_subset1.sum(1), axis=0)

