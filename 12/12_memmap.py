# -*- coding: utf-8 -*-
from __future__ import division
from numpy.random import randn
from pandas import Series
import numpy as np
np.set_printoptions(precision=4)
import sys

data = np.arange(12, dtype='float32')
data.resize((3,4))

from tempfile import mkdtemp
import os.path as path
filename = path.join(mkdtemp(), 'newfile.dat')  #建立缓存文件

fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))

fp[:] = data[:]
fp

fp.filename == path.abspath(filename)

del fp

newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))

fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
fpc.flags.writeable

fpc[0,:] = 0


fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
