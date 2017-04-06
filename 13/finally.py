# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 11:59:52 2016

@author: MJ
"""

f=open(path, 'w')

try:
    write_to_file(f)
except:
    print 'Failed'
else:
    print 'Succeeded'
finally:
    f.close()
