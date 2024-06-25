from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random
import json


class UnityJointType(str, Enum):
    FIXED = "FIXED"
    FREE = "FREE"
    SLIDE = "SLIDE"
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
class UnityData:
    pass


@dataclass
class UnityAsset(UnityData):
    tag: str


@dataclass
class UnityMesh(UnityAsset):
    dataID: str
    # (offset: bytes, count: int)
    indicesLayout: Tuple[int, int]
    normalsLayout: Tuple[int, int]
    verticesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]
    type: UnityAssetType = UnityAssetType.MESH


@dataclass
class UnityMaterial(UnityAsset):
    color: List[float]  # All the color values are within the 0 - 1.0 range
    emissionColor: List[float]
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0
    texture: Optional[str] = None
    texsize: Tuple[int, int] = (1, 1)
    type: UnityAssetType = UnityAssetType.MATERIAL


@dataclass
class UnityTexture(UnityAsset):
    dataID: str
    width: int = 0
    height: int = 0
    textype: str = "cube"
    type: UnityAssetType = UnityAssetType.TEXTURE


@dataclass
class UnityTransform(UnityData):
    pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    rot: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    scale: List[float] = field(default_factory=lambda: [1, 1, 1])

    def __add__(self, other: UnityTransform):
        pos = np.array(self.pos) + np.array(other.pos)
        rot = np.array(self.rot) + np.array(other.rot)
        scale = np.array(self.scale) * np.array(other.scale)
        return UnityTransform(
            pos=pos.tolist(),
            rot=rot.tolist(),
            scale=scale.tolist(),
        )


@dataclass
class UnityVisual(UnityData):

    type: UnityVisualType
    trans: UnityTransform
    mesh: str = None
    material: str = None
    color: List[float] = None


@dataclass
class UnityJoint(UnityData):
    name: str
    type: UnityJointType = UnityJointType.FIXED
    initial: float = 0.0
    maxrot: float = 0.0
    minrot: float = 0.0
    trans: UnityTransform = field(default_factory=UnityTransform)
    axis: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class SimObject(UnityData):
    name: str
    trans: UnityTransform = field(default_factory=UnityTransform)
    joint: UnityJoint = None
    visuals: List[UnityVisual] = field(default_factory=list)
    children: List[SimObject] = field(default_factory=list)


class UnityScene(UnityData):

    def __init__(self) -> None:
        self.root: SimObject = None
        self.id: str = str(random.randint(int(1e9), int(1e10 - 1)))
        self.meshes: List[UnityMesh] = list()
        self.textures: List[UnityMesh] = list()
        self.materials: List[UnityMesh] = list()
        self.raw_data: Dict[str, bytes] = dict()

    def to_string(self) -> str:
        dict_data = {
            "root": asdict(self.root),
            "id": self.id,
            "meshes": [asdict(mesh) for mesh in self.meshes],
            "textures": [asdict(tex) for tex in self.textures],
            "materials": [asdict(mat) for mat in self.materials],
        }
        return json.dumps(dict_data)
