from xml.etree.ElementTree import Element as XMLNode

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance



def get_name(node: XMLNode, default_name: str = "default") -> str:
    return node.get("name", default_name)

def get_pos(node: XMLNode) -> list[float]:
    return list(map(float, node.get("pos", "0.0 0.0 0.0").split()))

def get_rot(node: XMLNode) -> list[float]:
    return list(map(float, node.get("rot", "0.0 0.0 0.0").split()))