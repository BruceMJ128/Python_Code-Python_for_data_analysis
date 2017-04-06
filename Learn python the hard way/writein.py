from sys import argv

script, filename = argv

txt = open(filename, 'w')

input = raw_input("Input info: ")

txt.write(input)

txt2=open(filename)

print txt2.read()

txt.close()