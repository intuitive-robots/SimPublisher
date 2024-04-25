from xml.etree.ElementTree import Element as XMLNode
from typing import List

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def ros2unity(array: List[float]):
    return [-array[1], array[2], array[0]]

def mj2unity_pos(pos):
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat):
    # note that the order is "[x, y, z, w]"
    return [quat[2], -quat[3], -quat[1], quat[0]]


def unity2mj_pos(pos):
    return [-pos[1], pos[2], pos[0]]


def unity2mj_quat(quat):
    # note that the order is "[x, y, z, w]"
    return [quat[2], -quat[3], -quat[1], quat[0]]

def extract_array_from_xml(node: XMLNode, key: str, default: str = "0.0 0.0 0.0") -> list[float]:
    return [float(x) for x in node.get(key, default).split()]