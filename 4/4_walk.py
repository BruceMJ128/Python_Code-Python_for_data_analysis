import numpy as np; import pandas as pd; import pylab as pl
from pandas import DataFrame, Series

import random

position =0
walk = [position]
steps = 1000

dict_walk = {}

for i in xrange(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
    #dict_walk[i] = position
    
#DF_walk = pd.DataFrame(dict_walk.item())
DF_walk = pd.DataFrame(walk)

DF_walk.plot()

loc=(np.abs(walk)>=10).argmax()

''' 
#different way to realize walk
nsteps = 1000
draws = np.random.randint(0,2,size = nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
'''

# 5000 circles of random walk
nwalks = 5000
nsteps = 1000
draws_c = np.random.randint(0,2,size=(nwalks,nsteps))
steps_c = np.where(draws_c>0, 1,-1)
walks_c = steps_c.cumsum(1) #axis =1

hits30 = (np.abs(walks_c)>=30).any(1)

crossing_times = (np.abs(walks_c[hits30]) >=30).argmax(1)