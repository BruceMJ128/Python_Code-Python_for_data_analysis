import re
def clean_strings(strings):
    result=[]
    for value in strings:
        value=value.strip()
        value=re.sub('[!#?]', '', value)
        value=value.title()
        result.append(value)
    return result