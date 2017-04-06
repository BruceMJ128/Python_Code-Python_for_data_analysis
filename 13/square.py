def squares(n=10):
    print 'Generating squares from 1 to %d' % (n**2)
    for i in xrange(1, n+1):
        yield i**2