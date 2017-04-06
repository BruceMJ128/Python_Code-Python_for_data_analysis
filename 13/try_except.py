# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:54:20 2016

@author: MJ
"""
def attempt_float(x):
    try:
        return float(x)
    except:
        print 'Failed'
    else:
        print 'Succeeded'
   

    