from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import shapefile

#from shapelib import ShapeFile
#import dbfilb

import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax = fig.add_subplot(349)
#ax.plot(x,y)
plt.show()

import numpy as np

x = np.array([1, 2, 3])
y = x
z = np.copy(x)

x[0] = 10
x[0] == y[0]
x[0] == z[0]