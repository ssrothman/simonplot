from typing import List, Union

def ensure_same_length(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append(arg)
        else:
            result.append([arg])
    
    maxlen = max([len(x) for x in result])

    for i in range(len(result)):
        if len(result[i]) == 1:
            result[i] = result[i] * maxlen
        elif len(result[i]) != maxlen:
            raise ValueError("All input arguments must have the same length or be of length 1")

    return result


def all_same_key(things : List, skip : Union[int, None]=None):
    
    indices = list(range(len(things)))
    if skip is not None:
        indices.remove(skip)

    if len(indices) == 0:
        return True

    thing0 = things[indices[0]]

    for i in indices[1:]:
        if thing0.key != things[i].key:
            return False
        
    return True
