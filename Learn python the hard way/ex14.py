from sys import argv

script, user_name, hobby = argv
prompt = '>'

print "Hi %s, I'm the %s script." %(user_name, script)
print "I'd like to ask you a few questions."
print "Do you like me %s?" %user_name
likes = raw_input(prompt)

print "Where do you live %s?" %user_name
lives = raw_input(prompt)

print "What kind of computer do you have?" 
computer = raw_input(prompt)

print "Your hobby is %r." %hobby
hobby_time = raw_input("You start to play "+ hobby +" from: ")

print """
ALright, so you said %r about liking me.
You live in %r. Not sure where that is.
And you have a %r computer. Nice.
""" % (likes, lives, computer)

print """
You begin to play %r from %r. 
""" % (hobby, hobby_time)
