from json import dumps
from typing import List

def dict2encodedstr(d: dict):
    return str.encode(dumps(d))

def string2floatlist(s: str):
    return [ord(c) for c in list(s)]

def floatlist2string(l: List[float]):
    return map(lambda x: chr(int(x)), l)

