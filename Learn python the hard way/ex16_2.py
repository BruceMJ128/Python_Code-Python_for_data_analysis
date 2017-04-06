from sys import argv

script, filename = argv

x = open(filename)

print x.read()
x.close()