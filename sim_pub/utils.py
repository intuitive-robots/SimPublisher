from json import dumps

def dict2encodedstr(d: dict):
    return str.encode(dumps(d))