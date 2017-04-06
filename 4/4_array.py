import numpy as np; import pandas as pd

from numpy import random
from numpy import ix_



arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])

data = np.random.randn(7, 4)

arr = np.empty((8,4))
for i in range(8):
    arr[i] = i+10
    
arr2 = np.arange(32).reshape((8,4))
arr3 = arr2[np.ix_([1,5,7,2],[0,3,1,2])]

arr4 = np.arange(18).reshape((6,3))

x_doc = np.dot(arr4.T, arr4)

x = np.array([1,2,3])
y = np.array([5,6,7])

t1 = np.modf([1.2,-3.4])

xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])

result = np.where(cond, xarr, yarr)

arr = np.random.randn(4,4)

r1 = np.random.randn(1000,1)

from numpy.linalg import inv, qr

x = np.arange(25).reshape((5,5))

mat = x.T.dot(x)
