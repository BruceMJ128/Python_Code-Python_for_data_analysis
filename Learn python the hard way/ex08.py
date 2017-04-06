formatter = "%r %r %r %r"

print formatter % (1,2 ,3 ,4)
print formatter % ("One", "two", "three", "four")
print formatter % (True, False, False, True)
print formatter % (formatter, formatter, formatter, formatter)
print formatter % (
  "I had this thing.",
  "That you could type up right.",
  "But it didn't sing.",
  "So I said goodnight."
  )
  
print formatter % ("a apple", "beynod yourself", "control life", "didn't")

print "word %r" % "didn't"
print "word %r" % "did not"

print "word %s" % "didn't"
print "word %s" % "did not"