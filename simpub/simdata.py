from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random
import json


class VisualType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    CYLINDER = "CYLINDER"
    PLANE = "PLANE"
    QUAD = "QUAD"
    MESH = "MESH",
    NONE = "NONE"


class AssetType(str, Enum):
    MATERIAL = "MATERIAL",
    TEXTURE = "TEXTURE",
    MESH = "MESH",


@dataclass
class SimData:
    pass


@dataclass
class SimAsset(SimData):
    tag: str


@dataclass
class SimMesh(SimAsset):
    dataID: str
    # (offset: bytes, count: int)
    indicesLayout: Tuple[int, int]
    normalsLayout: Tuple[int, int]
    verticesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]
    type: AssetType = AssetType.MESH


@dataclass
class SimMaterial(SimAsset):
    color: List[float]  # All the color values are within the 0 - 1.0 range
    emissionColor: List[float]
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0
    texture: Optional[str] = None
    texsize: Tuple[int, int] = (1, 1)
    type: AssetType = AssetType.MATERIAL


@dataclass
class SimTexture(SimAsset):
    dataID: str
    width: int = 0
    height: int = 0
    textype: str = "cube"
    type: AssetType = AssetType.TEXTURE


@dataclass
class SimTransform(SimData):
    pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    rot: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    scale: List[float] = field(default_factory=lambda: [1, 1, 1])

    def __add__(self, other: SimTransform):
        pos = np.array(self.pos) + np.array(other.pos)
        rot = np.array(self.rot) + np.array(other.rot)
        scale = np.array(self.scale) * np.array(other.scale)
        return SimTransform(
            pos=pos.tolist(),
            rot=rot.tolist(),
            scale=scale.tolist(),
        )


@dataclass
class SimVisual(SimData):

    type: VisualType
    trans: SimTransform
    mesh: str = None
    material: str = None
    color: List[float] = None


@dataclass
class SimObject(SimData):
    name: str
    trans: SimTransform = field(default_factory=SimTransform)
    visuals: List[SimVisual] = field(default_factory=list)
    children: List[SimObject] = field(default_factory=list)


class SimScene(SimData):

    def __init__(self) -> None:
        self.root: SimObject = None
        self.id: str = str(random.randint(int(1e9), int(1e10 - 1)))
        self.meshes: List[SimMesh] = list()
        self.textures: List[SimMesh] = list()
        self.materials: List[SimMesh] = list()
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
