from typing import List, Dict, Callable
import numpy as np
from scipy.spatial.transform import Rotation
from xml.etree.ElementTree import Element as XMLNode
import re

from simpub.unity_data import UnityVisualType

RMap: Dict[str, Callable] = {
    "quat": lambda x: quat2quat(x),
    "axisangle": lambda x: axisangle2quat(x),
    "euler": lambda x: euler2quat(x),
    "xyaxes": lambda x: xyaxes2quat(x),
    "zaxis": lambda x: zaxis2quat(x),
}


def get_rot_from_xml(obj_xml: XMLNode) -> List[float]:
    result: List[float] = [0, 0, 0, 1]
    for key in RMap.keys():
        if key in obj_xml.attrib:
            result = RMap[key](
                str2list(obj_xml.get(key))
            )
            break
    return ros2unity_quat(result)
    # return [-result[0], result[1], result[2]]


def str2list(input_str: str) -> List[float]:
    return [float(num) for num in re.split(r'[ ,\n]+', input_str)]


def str2listabs(input_str: str, sep: str = ' ') -> List[float]:
    return [abs(float(num)) for num in input_str.split(sep)]


def rotation2unity(rotation: Rotation) -> List[float]:
    print(rotation.as_quat().tolist())
    return rotation.as_quat().tolist()


def quat2quat(quat: List[float]) -> List[float]:
    quat = np.asarray(quat, dtype=np.float32)
    assert len(quat) == 4, "Quaternion must have four components."
    # Mujoco use wxyz format and Unity uses xyzw format
    w, x, y, z = quat
    quat = np.array([x, y, z, w], dtype=np.float64)
    return rotation2unity(Rotation.from_quat(quat))


def axisangle2quat(
    axisangle: List[float], use_degree=True
) -> List[float]:
    assert len(axisangle) == 4, (
        "axisangle must contain four values (x, y, z, a)."
    )
    # Extract the axis (x, y, z) and the angle a
    axis = axisangle[:3]
    angle = axisangle[3]
    axis = axis / np.linalg.norm(axis)
    if use_degree:
        angle = np.deg2rad(angle)
    rotation = Rotation.from_rotvec(angle * axis)
    return rotation2unity(rotation)


def euler2quat(
    euler: List[float], degree: str = True
) -> List[float]:
    assert len(euler) == 3, "euler must contain three values (x, y, z)."
    # Convert the Euler angles to radians if necessary
    if not degree:
        euler = np.rad2deg(euler).tolist()
    return euler


def xyaxes2quat(xyaxes: List[float]) -> List[float]:
    assert len(xyaxes) == 6, (
        "xyaxes must contain six values (x1, y1, z1, x2, y2, z2)."
    )
    x = np.array(xyaxes[:3])
    y = np.array(xyaxes[3:])
    z = np.cross(x, y)
    rotation_matrix = np.array([x, y, z]).T
    rotation = Rotation.from_matrix(rotation_matrix)
    return rotation2unity(rotation)


def zaxis2quat(zaxis: List[float]) -> List[float]:
    assert len(zaxis) == 3, "zaxis must contain three values (x, y, z)."
    # Create the rotation object from the z-axis
    rotation = Rotation.from_rotvec(np.pi, np.array(zaxis))
    return rotation2unity(rotation)


def ros2unity(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def ros2unity_quat(quat: List[float]) -> List[float]:
    return [quat[1], -quat[2], -quat[0], quat[3]]


TypeMap: Dict[str, UnityVisualType] = {
    "plane": UnityVisualType.PLANE,
    "sphere": UnityVisualType.SPHERE,
    "capsule": UnityVisualType.CAPSULE,
    "ellipsoid": UnityVisualType.CAPSULE,
    "cylinder": UnityVisualType.CYLINDER,
    "box": UnityVisualType.CUBE,
    "mesh": UnityVisualType.MESH
}


def scale2unity(scale: List[float], visual_type: str) -> List[float]:
    if visual_type in ScaleMap:
        return ScaleMap[visual_type](scale)
    else:
        return [1, 1, 1]


def plane2unity_scale(scale: List[float]) -> List[float]:
    return list(map(abs, [scale[0] * 2, 0.001, scale[1] * 2]))


def box2unity_scale(scale: List[float]) -> List[float]:
    # return [abs(scale[1]) * 2, abs(scale[2]) * 2, abs(scale[0]) * 2]
    return [abs(scale[i]) * 2 for i in [1, 2, 0]]


def sphere2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[0]) * 2] * 3


def cylinder2unity_scale(scale: List[float]) -> List[float]:
    if len(scale) == 3:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    else:
        return list(map(abs, [scale[0] * 2, scale[1], scale[0] * 2]))


def capsule2unity_scale(scale: List[float]) -> List[float]:
    assert len(scale) == 3, "Only support scale with three components."
    return list(map(abs, [scale[0], scale[1], scale[0]]))


ScaleMap: Dict[str, Callable] = {
    "plane": lambda x: plane2unity_scale(x),
    "box": lambda x: box2unity_scale(x),
    "sphere": lambda x: sphere2unity_scale(x),
    "cylinder": lambda x: cylinder2unity_scale(x),
    "capsule": lambda x: capsule2unity_scale(x),
    "ellipsoid": lambda x: capsule2unity_scale(x),
}