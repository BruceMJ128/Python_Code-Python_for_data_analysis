# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import shapefile

sf = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020')
shp = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020.shp')
dbf = shapefile.Reader('data/ch09/statesp020_nt00032/statesp020.dbf')

'''
sf的数据结构如下：
dir(sf)

Out[68]: 
['_Reader__dbfHdrLength',
 '_Reader__dbfHeader',
 '_Reader__dbfHeaderLength',
 '_Reader__getFileObj',
 '_Reader__record',
 '_Reader__recordFmt',
 '_Reader__restrictIndex',
 '_Reader__shape',
 '_Reader__shapeIndex',
 '_Reader__shpHeader',
 '__doc__',
 '__init__',
 '__module__',
 '_offsets',
 'bbox',
 'dbf',
 'elevation',
 'fields',
 'iterRecords',
 'iterShapes',
 'load',
 'measure',
 'numRecords',
 'record',
 'records',
 'shape',
 'shapeName',
 'shapeRecord',
 'shapeRecords',
 'shapeType',
 'shapes',
 'shp',
 'shpLength',
 'shx']

'''

myshp = open('data/ch09/statesp020_nt00032/statesp020.shp', "rb")
mydbf = open('data/ch09/statesp020_nt00032/statesp020.dbf', "rb")
r = shapefile.Reader(shp=myshp, dbf=mydbf)

shapes = sf.shapes() #instance.shapes(), shapes is list of instance

len(shapes)

len(list(sf.iterShapes())) #type of sf.iterShapes() is generator, means iterator

#list(sf.iterShapes()) 等效于 sf.shapes()


for name in dir(shapes[3]):  #type of dir(shapes[3]) is list, dir()查看shapes[3]的功能列表
    if not name.startswith('__'):
        name
'bbox'
'parts'
'points'
'shapeType'

shapes[3].shapeType  #instance.shapeType is int

# Get the bounding box of the 4th shape.
# Round coordinates to 3 decimal places

bbox = shapes[3].bbox  #bbox type is list, [a,b,c,d]
['%.3f' % coord for coord in bbox]  

'''
['-155.575', '71.214', '-155.440', '71.239']
'''

shapes[3].parts #tyoe is list: [0]
len(shapes[3].points) #type of shapes[3].points is list: [[a1,b1],[a2,b2],...,[an,bn]]

'''
shapes[3].points[:2]  #应该讲的是这个州在地图上的每个节点
Out[4]: 
[[-155.44296264648438, 71.21398162841797],
 [-155.46722412109375, 71.22029876708984]]
'''

# Get the 8th point of the fourth shape
# Truncate coordinates to 3 decimal places

shape = shapes[3].points[7] #[ax,bx]
['%.3f' % coord for coord in shape]

s = sf.shape(7) #等效于shapes = sf.shapes(), s = shapes[7]
# Read the bbox of the 8th shape to verify
# Round coordinates to 3 decimal places
['%.3f' % coord for coord in s.bbox]  



fields = sf.fields #type is list: [(s1,s2,n1,n2),[s3,s4,n4,n5],[..],...,[sx,sx2,nx,nx2]]

'''
fields[:2]
Out[19]: [('DeletionFlag', 'C', 1, 0), ['AREA', 'N', 12, 3]]
'''

records = sf.records() #list: [[n1,n2,n3,s4,s5,n6,s7,n8,s9],[..],...,[..]]

'''
records[0]
Out[18]: [267.357, 374.768, 2, 'Alaska', '02', 49, 'January', 3, 1959]
'''

len(records)

len(list(sf.iterRecords()))

records[3][1:3]

rec = sf.record(3) #等效于 records = sf.records(), rec = records[3]

rec[1:3]

shapeRecs = sf.shapeRecords()

shapeRecs[3].record[1:3]

points = shapeRecs[3].shape.points[0:2]

len(points)

shapeRec = sf.shapeRecord(3)

shapeRec.record[1:3]


points = shapeRec.shape.points[0:2] #shapeRec.shape等效于刚刚的sf.shapes()数据结构

len(points)

#shapeRecs = sf.iterShapeRecords()  #sf的数据结构中没有iterShapeRecords()这一项
#for shapeRec in shapeRecs: # do something here
#    pass

#写入 Shapefiles

w = shapefile.Writer() #type is instance

'''
w的数据结构
dir(w)

Out[86]: 
['_Writer__bbox',
 '_Writer__dbfHeader',
 '_Writer__dbfRecords',
 '_Writer__getFileObj',
 '_Writer__mbox',
 '_Writer__shapefileHeader',
 '_Writer__shpFileLength',
 '_Writer__shpRecords',
 '_Writer__shxRecords',
 '_Writer__zbox',
 '__doc__',
 '__init__',
 '__module__',
 '_lengths',
 '_offsets',
 '_shapes',
 'bbox',
 'dbf',
 'deletionFlag',
 'field',
 'fields',
 'line',
 'mbox',
 'null',
 'point',
 'poly',
 'record',
 'records',
 'save',
 'saveDbf',
 'saveShp',
 'saveShx',
 'shape',
 'shapeType',
 'shapes',
 'shp',
 'shx',
 'zbox']

'''

w = shapefile.Writer(shapeType=1)
w.shapeType



w.shapeType = 3

w.shapeType

w.autoBalance = 1

w = shapefile.Writer()

w.point(122, 37) # No elevation or measure values

w.shapes()[0].points #[[1, 5, 0, 0], [5, 5, 0, 0], [5, 1, 0, 0], [3, 3, 0, 0], [1, 1, 0, 0]]

w.point(118, 36, 4, 8)

w.shapes()[1].points #[[122, 37, 0, 0]]
#w.shapes()[2].points #Out[116]: [[118, 36, 4, 8]]


w = shapefile.Writer()

w.poly(shapeType=3, parts=[[[122,37,4,9], [117,36,3,4]], [[115,32,8,8],[118,20,6,4], [113,24]]])

#w.poly()-->polygons多边形

w = shapefile.Writer()

w.null()

#assert w.shapes()[0].shapeType == shapefile.NULL

#----Creating Attributes---

w = shapefile.Writer(shapefile.POINT)
w.point(1,1)

w.point(3,1)
w.point(4,3)
w.point(2,2)
w.field('FIRST_FLD')
w.field('SECOND_FLD','C','40')
w.record('First','Point')
w.record('Second','Point')
w.record('Third','Point')
w.record('Fourth','Point')
w.save('data/ch09/shapefiles/test/point')

w = shapefile.Writer(shapefile.POLYGON)
w.poly(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record('First','Polygon')
w.save('data/ch09/shapefiles/test/polygon')

w = shapefile.Writer(shapefile.POLYLINE) #POLYLINE 折线
w.line(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
w.poly(parts=[[[1,3],[5,3]]], shapeType=shapefile.POLYLINE)
w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record('First','Line')
w.record('Second','Line')
w.save('data/ch09/shapefiles/test/line')


w = shapefile.Writer(shapefile.POLYLINE)
w.line(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record(FIRST_FLD='First', SECOND_FLD='Line')
w.save('data/ch09/shapefiles/test/line') #自动保存为 line.dbf, line.shp, line.shx三个文件

# --------------File Names--------------

targetName = w.save()
assert("shapefile_" in targetName)

# ------Saving to File-Like Objects------

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
shp = StringIO()
shx = StringIO()
dbf = StringIO()
w.saveShp(shp)
w.saveShx(shx)
w.saveDbf(dbf)
# Normally you would call the "StringIO.getvalue()" method on these objects.
shp = shx = dbf = None

# -------------Editing Shapefiles---------------

e = shapefile.Editor(shapefile="data/ch09/shapefiles/test/point.shp")
e.point(0,0,10,2)
e.record("Appended","Point")
e.save('data/ch09/shapefiles/test/point')


e = shapefile.Editor(shapefile="data/ch09/shapefiles/test/line.shp")
e.line(parts=[[[10,5],[15,5],[15,1],[13,3],[11,1]]])
e.record('Appended','Line')
e.save('data/ch09/shapefiles/test/line')


e = shapefile.Editor(shapefile="data/ch09/shapefiles/test/polygon.shp")
e.poly(parts=[[[5.1,5],[9.9,5],[9.9,1],[7.5,3],[5.1,1]]])
e.record("Appended","Polygon")
e.save('data/ch09/shapefiles/test/polygon')

#Remove the first point in each shapefile - for a point shapefile that is the first shape and record”

e = shapefile.Editor(shapefile="data/ch09/shapefiles/test/point.shp")
e.delete(0)
e.save('data/ch09/shapefiles/test/point')


e = shapefile.Editor(shapefile="data/ch09/shapefiles/test/polygon.shp")
e.delete(-1)
e.save('data/ch09/shapefiles/test/polygon')

# ------------Python __geo_interface__--------------

s = sf.shape(0)
s.__geo_interface__["type"]
'MultiPolygon'

#Testing 运行

#$ python shapefile.py

