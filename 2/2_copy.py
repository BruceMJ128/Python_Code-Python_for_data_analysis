import copy
l1 = [[1,2],['a','b','c']]

l2= copy.copy(l1)
print l2
print l1

print '\n'

l2[0].append(10)
print l2
print l1

l3 = copy.deepcopy(l1)

l3[0].append(30)

print l3
print l1
