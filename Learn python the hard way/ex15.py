from sys import argv #mark import

script, filename = argv #define argv

txt = open(filename) #put the filename location to a local varials txt

print "Here's your file %r:" %filename
print txt.readline()
print txt.read() #read the content of variable txt

print "Type the filename again:"
file_again = raw_input("> ")

txt_again = open(file_again)

print txt_again.read()


txt.close()
txt_again.close()

