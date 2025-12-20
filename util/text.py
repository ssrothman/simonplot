import re

def strip_units(s: str) -> str:
    if '[' in s and ']' in s:
        start = s.index('[')
        end = s.index(']')
        return s[:start].strip() + s[end+1:].strip()
    else:
        return s.strip()
    
def strip_dollar_signs(s: str) -> str:
    return s.replace('$', '')

def attempt_regex_match(pattern: str, axiskey: str) -> bool:
    #escape special characters
    pattern_escaped = re.escape(pattern)
    #replace '*' wildcard with alphanumeric match
    pattern_escaped = pattern_escaped.replace(r'\*', r'[a-zA-Z0-9]*')
    #ensure no leading or trailing characters
    pattern_escaped = "^" + pattern_escaped + "$" 

    return re.fullmatch(pattern_escaped, axiskey) is not None
