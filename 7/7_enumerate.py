# -*- coding: utf-8 -*-
import pandas as pd; import numpy as np

from pandas import DataFrame

from pandas import Series

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/ch02/movielens/movies.dat', sep='::', header=None,
                        names=mnames)
genre_iter = (set(x.split('|')) for x in movies.genres) #type is genetator
genres = sorted(set.union(*genre_iter)) #type is list, genetator 转化为 genre_iter

dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)


for i, gen in enumerate(movies.genres): 
    dummies.ix[i, gen.split('|')] = 1
# i=1, gen='Drama|Thriller'
# gen.split('|')=['Drama', 'Thriller']

'''
dummies.ix[1]
Out[15]: 
Action         0.0
Adventure      1.0
Animation      0.0
Children's     1.0
Comedy         0.0
Crime          0.0
Documentary    0.0
Drama          0.0
Fantasy        1.0
Film-Noir      0.0
Horror         0.0
Musical        0.0
Mystery        0.0
Romance        0.0
Sci-Fi         0.0
Thriller       0.0
War            0.0
Western        0.0
Name: 1, dtype: float64

dummies.ix[1,gen.split('|')]  #赋值为1之后，Drama和Thriller都变为1
Out[14]: 
Drama       0.0
Thriller    0.0
Name: 1, dtype: float64

'''

movies_windic = movies.join(dummies.add_prefix('Genre_')) #在dummies的列头之前加‘Genew_’,并将dummies与movies合并
movies_windic.ix[0]

