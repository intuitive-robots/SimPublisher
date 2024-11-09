from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random
import json

# TODO: Using dict not dataclass to store for too many never-used attributes


class VisualType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    CYLINDER = "CYLINDER"
    PLANE = "PLANE"
    QUAD = "QUAD"
    MESH = "MESH",
    NONE = "NONE"


@dataclass
class SimData:
    pass


@dataclass
class SimAsset(SimData):
    hash: str


@dataclass
class SimMesh(SimAsset):
    # (offset: bytes, count: int)
    indicesLayout: Tuple[int, int]
    normalsLayout: Tuple[int, int]
    verticesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]


@dataclass
class SimMaterial(SimData):
    # All the color values are within the 0 - 1.0 range
    color: List[float]
    emissionColor: List[float]
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    texture: Optional[SimTexture] = None


@dataclass
class SimTexture(SimAsset):
    width: int = 0
    height: int = 0
    # TODO: add new texture type
    textureType: str = "2D"
    textureSize: Tuple[int, int] = field(default_factory=lambda: (1, 1))


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
    name: str
    type: VisualType
    trans: SimTransform
    material: Optional[SimMaterial] = None
    mesh: Optional[SimMesh] = None
    # TODO： easily set up transparency
    # def setup_transparency(self):
    #     if self.material is not None:
    #         self.material = self


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
        self.raw_data: Dict[str, bytes] = dict()

    def to_string(self) -> str:
        dict_data = {
            "root": asdict(self.root),
            "id": self.id,
        }
        return json.dumps(dict_data)
