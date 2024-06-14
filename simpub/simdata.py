from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random


class UnityJointType(str, Enum):
    FIXED = "FIXED"  # Fixed joint for unmovable objects
    FREE = "FREE"  # Free joint for free movement for one single body
    SLIDE = "SLIDE"  # Slide joint for linear movement
    BALL = "BALL"
    HINGE = "HINGE"


class UnityVisualType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    CYLINDER = "CYLINDER"
    PLANE = "PLANE"
    QUAD = "QUAD"
    MESH = "MESH",
    NONE = "NONE"


class UnityAssetType(str, Enum):
    MATERIAL = "MATERIAL",
    TEXTURE = "TEXTURE",
    MESH = "MESH",


@dataclass
class SimAsset:
    tag: str

    def __repr__(self):
        return f"<{type(self)} tag={self.tag}>"


@dataclass
class UnityMesh(SimAsset):
    dataID: str
    # (offset: bytes, count: int)
    indicesLayout: Tuple[int, int]
    normalsLayout: Tuple[int, int]
    verticesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]
    type: UnityAssetType = UnityAssetType.MESH


@dataclass
class UnityMaterial(SimAsset):
    # All the color values are within the 0 - 1.0 range
    color: np.ndarray
    emissionColor: np.ndarray
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0
    texture: Optional[str] = None
    texsize: Tuple[int, int] = (1, 1)
    type: UnityAssetType = UnityAssetType.MATERIAL


@dataclass
class UnityTexture(SimAsset):
    dataID: str
    width: int = 0
    height: int = 0
    textype: str = "cube"
    type: UnityAssetType = UnityAssetType.TEXTURE


"""
Scene data
"""
@dataclass
class UnityTransform:
    position: np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones((3, ), dtype=np.float32))

    def __add__(self, other: UnityTransform):
        return UnityTransform(
            position=self.position + other.position,
            rotation=self.rotation + other.rotation,
            scale=self.scale * other.scale
        )

    def __repr__(self) -> str:
        return f"<UnityTransform pos={self.position.tolist()} rot={self.rotation.tolist()} scale={self.scale.tolist()}>"


@dataclass
class UnityVisual:
    type: str = ""
    mesh: str = ""
    material: str = ""
    transform: UnityTransform
    color: List[float]

    def __repr__(self) -> str:
        return f"<UnityVisual tupe={self.type}>"


@dataclass
class UnityJoint:
    name: str
    type: UnityJointType = UnityJointType.FIXED
    initial: float = 0.0
    maxrot: float = 0.0
    minrot: float = 0.0
    transform: UnityTransform = field(default_factory=UnityTransform)
    axis: List[float] = field(default_factory=lambda: np.array([0, 0, 0]))

    def __repr__(self) -> str:
        return f"<UnityJoint name={self.name} type={self.type}>"


@dataclass
class UnityGameObject:
    name: str
    joint: UnityJoint
    visuals: List[UnityVisual]
    children: List[UnityGameObject]

    def get_joints(self, *types) -> List[UnityJoint]:
        select: set = set(types) if len(types) > 0 else set(UnityJointType)
        joints = list()
        found = [self]
        while found and (current := found.pop()):
            connected_joints = current.joint
            joints.extend(joint for joint in connected_joints if joint.type in select)
            found.extend(current.bodies)
        return joints

    def __repr__(self) -> str:
        return f"<UnityGameObject visuals={len(self.visuals)} " \
               f"joint={len(self.joint)} " \
               f"children={len(self.children)}>"


@dataclass
class UnityScene:
    worldbody: UnityGameObject = None
    id: str = str(random.randint(int(1e9), int(1e10 - 1)))
    meshes: List[UnityMesh] = field(default_factory=list)
    textures: List[UnityMesh] = field(default_factory=list)
    materials: List[UnityMesh] = field(default_factory=list)
    # _meta_data: Dict[str, Any] = field(default_factory=dict)
    _raw_data: Dict[str, bytes] = field(default_factory=dict)