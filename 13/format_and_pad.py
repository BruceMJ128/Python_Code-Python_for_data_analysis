def format_and_pad(template, space):
    def formatter(x):
        return (template % x).rjust(space)
    return formatter