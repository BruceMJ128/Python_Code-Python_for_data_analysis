# -*- coding: utf-8 -*-
# Advanced NumPy

from __future__ import division
from numpy.random import randn
from pandas import Series
import numpy as np
np.set_printoptions(precision=4)
import sys

## ndarray object internals

### NumPy dtype hierarchy


ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)


np.float64.mro()

## Advanced array manipulation

### Reshaping arrays


arr = np.arange(8)
arr
arr.reshape((4, 2))


arr.reshape((4, 2)).reshape((2, 4))


arr = np.arange(15)
arr.reshape((5, -1))


other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape)


arr = np.arange(15).reshape((5, 3))
arr
arr.ravel()


arr.flatten()

### C vs. Fortran order


arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F')

### Concatenating and splitting arrays


arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)


np.vstack((arr1, arr2))
np.hstack((arr1, arr2))


from numpy.random import randn
arr = randn(5, 2)
arr
first, second, third = np.split(arr, [1, 3])
print first
print second
print third

#### Stacking helpers: 


arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = randn(3, 2)
np.r_[arr1, arr2]
np.c_[np.r_[arr1, arr2], arr]


np.c_[1:6, -10:-5]

### Repeating elements: tile and repeat

arr = np.arange(3)
arr.repeat(3)


arr.repeat([2, 3, 4])


arr = randn(2, 2)
arr
arr.repeat(2, axis=0)


arr.repeat([2, 3], axis=0)
arr.repeat([2, 3], axis=1)


arr
np.tile(arr, 2)


arr
np.tile(arr, (2, 1))
np.tile(arr, (3, 2))

### Fancy indexing equivalents: take and put


arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds]


arr.take(inds)
arr.put(inds, 42)
arr
arr.put(inds, [40, 41, 42, 43])
arr


inds = [2, 0, 2, 1]
arr = randn(2, 4)
arr
arr.take(inds, axis=1)

## Broadcasting

arr = np.arange(5)
arr
arr * 4


arr = randn(4, 3)
arr.mean(0)
demeaned = arr - arr.mean(0)
demeaned
demeaned.mean(0)


arr
row_means = arr.mean(1)
row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
demeaned.mean(1)

### Broadcasting over other axes

#arr - arr.mean(1)


#arr - arr.mean(1).reshape((4, 1))


arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]
arr_3d.shape


arr_1d = np.random.normal(size=3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]


arr = randn(3, 4, 5)
depth_means = arr.mean(2)  #按照轴axis=2进行groupby,求mean
depth_means
demeaned = arr - depth_means[:, :, np.newaxis]
demeaned.mean(2)


def demean_axis(arr, axis=0):
    means = arr.mean(axis)

    # This generalized things like [:, :, np.newaxis] to N dimensions
    indexer = [slice(None)] * arr.ndim
    indexer[axis] = np.newaxis
    return arr - means[indexer]

### Setting array values by broadcasting


arr = np.zeros((4, 3))
arr[:] = 5
arr


col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr
arr[:2] = [[-1.37], [0.509]]
arr

## Advanced ufunc usage

### Ufunc instance methods


arr = np.arange(10)
np.add.reduce(arr)
arr.sum()


np.random.seed(12346)


arr = randn(5, 5)
arr[::2].sort(1) # sort a few rows
arr[:, :-1] < arr[:, 1:]
np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)  #按照axis=1方向，将所有bool型元素，用and的方法判断，全为True则为True，出现一个False，则为False


arr = np.arange(15).reshape((3, 5))
np.add.accumulate(arr, axis=1)


arr = np.arange(3).repeat([1, 2, 2])
arr
np.multiply.outer(arr, np.arange(5))


result = np.subtract.outer(randn(3, 4), randn(5))
result.shape


arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])


arr = np.multiply.outer(np.arange(4), np.arange(5))
arr
np.add.reduceat(arr, [0, 2, 4], axis=1)

### Custom ufuncs


def add_elements(x, y):
    return x + y
add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))


add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))


arr = randn(10000)
#%timeit add_them(arr, arr)
#%timeit np.add(arr, arr)

## Structured and record arrays


dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
sarr


sarr[0]
sarr[0]['y']


sarr['x']

### Nested dtypes and multidimensional fields


dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
arr


arr[0]['x']


arr['x']


dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
data['x']
data['y']
data['x']['a']

### Why use structured arrays?

### Structured array manipulations: numpy.lib.recfunctions

## More about sorting


arr = randn(6)
arr.sort()
arr


arr = randn(3, 5)
arr
arr[:, 0].sort() # Sort first column values in-place
arr


arr = randn(5)
arr
np.sort(arr)
arr


arr = randn(3, 5)
arr
arr.sort(axis=1)
arr


arr[:, ::-1]

### Indirect sorts: argsort and lexsort


values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
indexer
values[indexer]


arr = randn(3, 5)
arr[0] = values
arr
arr[:, arr[0].argsort()]


first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
zip(last_name[sorter], first_name[sorter])

### Alternate sort algorithms


values = np.array(['2:first', '2:second', '1:first', '1:second', '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
indexer
values.take(indexer)

### numpy.searchsorted: Finding elements in a sorted array


arr = np.array([0, 1, 7, 12, 15])
arr.searchsorted(9)


arr.searchsorted([0, 8, 11, 16])


arr = np.array([0, 0, 0, 1, 1, 1, 1])
arr.searchsorted([0, 1])
arr.searchsorted([0, 1], side='right')


data = np.floor(np.random.uniform(0, 10000, size=50))
bins = np.array([0, 100, 1000, 5000, 10000])
data


labels = bins.searchsorted(data)
labels


Series(data).groupby(labels).mean()


np.digitize(data, bins)

## NumPy matrix class


X = np.array([[ 8.82768214, 3.82222409, -1.14276475, 2.04411587],
    [ 3.82222409, 6.75272284, 0.83909108, 2.08293758],
    [-1.14276475, 0.83909108, 5.01690521, 0.79573241],
    [ 2.04411587, 2.08293758, 0.79573241, 6.24095859]])
X[:, 0] # one-dimensional
y = X[:, :1] # two-dimensional by slicing
X
y


np.dot(y.T, np.dot(X, y))


Xm = np.matrix(X)
ym = Xm[:, 0]
Xm
ym
ym.T * Xm * ym


Xm.I * X

## Advanced array input and output


### Memory-mapped files

from tempfile import mkdtemp
import os.path as path
filename = path.join(mkdtemp(), 'newfile.dat')

mmap = np.memmap(filename, dtype='float64', mode='w+', shape=(10000, 10000))
mmap


section = mmap[:5]


section[:] = np.random.randn(5, 10000)
mmap.flush()
mmap
del mmap


mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 10000))
mmap


#%xdel mmap
#!rm mymmap

### HDF5 and other array storage options

## Performance tips

### The importance of contiguous memory


arr_c = np.ones((1000, 1000), order='C')
arr_f = np.ones((1000, 1000), order='F')
arr_c.flags
arr_f.flags
arr_f.flags.f_contiguous


#%timeit arr_c.sum(1)
#%timeit arr_f.sum(1)


arr_f.copy('C').flags


arr_c[:50].flags.contiguous
arr_c[:, :50].flags


#%xdel arr_c
#%xdel arr_f
#%cd ..

## Other speed options: Cython, f2py, C
'''

from numpy import ndarray, float64_t

def sum_elements(ndarray[float64_t] arr):
    cdef Py_ssize_t i, n = len(arr)
    cdef float64_t result = 0

    for i in range(n):
        result += arr[i]
    
    return result

from numpy cimport ndarray, float64_t

def sum_elements(ndarray[float64_t] arr):
    cdef Py_ssize_t i, n = len(arr)
    cdef float64_t result = 0
    
    for i in range(n):
        result += arr[i]
    
    return result
CloseExpandOpen in PagerClose


'''
