from __future__ import annotations

import abc
import io
import json
from dataclasses import dataclass, field, fields
from enum import Enum
from hashlib import md5
from typing import Dict, List, Optional, Tuple, TypedDict
import msgpack

from PIL import Image
import numpy as np
import trimesh


class SimTransform(TypedDict):
    pos: List[float]
    rot: List[float]
    scale: List[float]


class SimMesh(TypedDict):
    indices: bytes
    vertices: bytes
    normals: bytes
    uv: Optional[bytes]

class SimTexture(TypedDict):
    width: int
    height: int
    textureType: str
    textureScale: Tuple[float, float]
    textureData: bytes

class SimMaterial(TypedDict):
    color: List[float]
    emissionColor: List[float]
    specular: float
    shininess: float
    reflectance: float
    texture: Optional[SimTexture]

class SimVisual(TypedDict):
    name: str
    type: str
    mesh: Optional[SimMesh]
    material: Optional[SimMaterial]
    trans: SimTransform

class SimObject(TypedDict):
    name: str
    parent: str
    trans: SimTransform
    visuals: List[SimVisual]

class SimSceneSetting(TypedDict):
    name: str



class TreeNode:
    def __init__(self) -> None:
        self.data: Optional[SimObject] = None
        self.children: List[TreeNode] = []

    def add_data(self, data: SimObject) -> None:
        self.data = data

    def add_child(self, child_node: TreeNode) -> None:
        self.children.append(child_node)

# @dataclass
# class SimData:
#     def to_dict(self):
#         return {
#             f.name: getattr(self, f.name)
#             for f in fields(self)
#             if not f.metadata.get("exclude", False)
#         }

#     def serialize(self) -> bytes:
#         data = msgpack.packb(self.to_dict(), use_bin_type=True)
#         assert isinstance(data, bytes), "msgpack serialization failed"
#         return data

# @dataclass
# class SimTransform(SimData):
#     pos: List[float] = field(default_factory=lambda: [0, 0, 0])
#     rot: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
#     scale: List[float] = field(default_factory=lambda: [1, 1, 1])




# @dataclass
# class SimVisual(SimData):
#     name: str
#     type: VisualType
#     trans: SimTransform
#     mesh: Optional[SimMesh] = None
#     material: Optional[SimMaterial] = None
#     # TODO： easily set up transparency
#     # def setup_transparency(self):
#     #     if self.material is not None:
#     #         self.material = self

#     def to_dict(self) -> Dict:
#         return {
#             "name": self.name,
#             "type": self.type.value,
#             "trans": self.trans.to_dict(),
#             "mesh": self.mesh.to_dict() if self.mesh else None,
#             "material": self.material.to_dict() if self.material else None,
#         }




# @dataclass
# class SimMaterial(SimData):
#     # All the color values are within the 0 - 1.0 range
#     color: List[float]
#     emissionColor: Optional[List[float]] = None
#     specular: float = 0.5
#     shininess: float = 0.5
#     reflectance: float = 0.0
#     texture: Optional[SimTexture] = None

#     def to_dict(self):
#         data = super().to_dict()
#         if self.texture is not None:
#             data["texture"] = self.texture.to_dict()
#         return data


# @dataclass
# class SimMesh(SimData):
#     # (offset: bytes, count: int)
#     indices: bytes
#     vertices: bytes
#     normals: bytes
#     uv: Optional[bytes] = None


# @dataclass
# class SimTexture(SimData):
#     hash: str
#     width: int
#     height: int
#     # TODO: new texture type
#     textureType: str
#     textureScale: Tuple[int, int]
#     textureData: bytes


# @dataclass
# class SimObject(SimData):
#     name: str
#     trans: SimTransform = field(default_factory=SimTransform)
#     visuals: List[SimVisual] = field(default_factory=list)
#     children: List[SimObject] = field(default_factory=list)

#     def to_dict(self) -> Dict:
#         return {
#             "name": self.name,
#             "trans": self.trans.to_dict(),
#             "visuals": [visual.to_dict() for visual in self.visuals],
#         }

#     def find_child(self, name: str) -> Optional[SimObject]:
#         if self.name == name:
#             return self
#         for child in self.children:
#             result = child.find_child(name)
#             if result is not None:
#                 return result
#         return None

class SimScene:
    def __init__(self, name: str = "DefaultSceneName") -> None:
        self.root: TreeNode = TreeNode()
        self.setting: SimSceneSetting = SimSceneSetting(name=name)

    @property
    def name(self) -> str:
        return self.setting["name"]

    def process_sim_obj(self, kt_node: TreeNode) -> None:
        # for visual in sim_obj.visuals:
        #     if visual.mesh is not None:
        #         visual.mesh.generate_normals(self.raw_data)
        for child in kt_node.children:
            self.process_sim_obj(child)

class VisualType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    CYLINDER = "CYLINDER"
    PLANE = "PLANE"
    QUAD = "QUAD"
    MESH = "MESH"
    NONE = "NONE"

def create_mesh(trimesh_obj: trimesh.Trimesh, uvs: Optional[np.ndarray]) -> SimMesh:
    trimesh_obj.fix_normals()
    vertices = trimesh_obj.vertices
    indices = trimesh_obj.faces
    normals = trimesh_obj.vertex_normals
    # Vertices
    vertices = vertices.astype(np.float32)
    num_vertices = vertices.shape[0]
    vertices = vertices[:, [1, 2, 0]]
    vertices[:, 0] = -vertices[:, 0]
    vertices = vertices.flatten()
    # Indices / faces
    indices = indices.astype(np.int32)
    indices = indices[:, [2, 1, 0]]
    indices = indices.flatten()
    # Normals
    normals = normals.astype(np.float32)
    normals = normals[:, [1, 2, 0]]
    normals[:, 0] = -normals[:, 0]
    normals = normals.flatten()
    assert normals.size == num_vertices * 3, (
        f"Number of vertex normals ({normals.shape[0]}) must be equal "
        f"to number of vertices ({num_vertices})"
    )
    assert np.max(indices) < num_vertices, (
        f"Index value exceeds number of vertices: {np.max(indices)} >= "
        f"{num_vertices}"
    )
    assert (
        indices.size % 3 == 0
    ), f"Number of indices ({indices.size}) must be a multiple of 3"
    return SimMesh(
        vertices=generate_bytes(vertices),
        indices=generate_bytes(indices),
        normals=generate_bytes(normals),
        uv=generate_bytes(uvs) if uvs is not None else None,
    )

def create_texture(
    image_flaten_array: np.ndarray,
    image_height: int,
    image_width: int,
    texture_scale: Tuple[int, int] = field(default_factory=lambda: (1, 1)),
    texture_type: str = "2D",
    quality: int = 75,
) -> SimTexture:
    image_array = image_flaten_array.reshape((image_height, image_width, -1))
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    # compress the image to JPEG format
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    image_bytes = buffer.getvalue()
    return SimTexture(
        width=image_array.shape[1],
        height=image_array.shape[0],
        textureType=texture_type,
        textureScale=texture_scale,
        textureData=image_bytes,
    )



def generate_raw_data(self) -> bytes:
    raise NotImplementedError

def generate_hash(data: bytes) -> str:
    return md5(data).hexdigest()

def generate_bytes(data: np.ndarray) -> bytes:
    # change all float np array to float32 and all int np array to int32
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    elif data.dtype == np.int64:
        data = data.astype(np.int32)
    return data.tobytes()

def create_material(
    color: List[float],
    emissionColor: Optional[List[float]] = None,
    specular: float = 0.5,
    shininess: float = 0.5,
    reflectance: float = 0.0,
    texture: Optional[SimTexture] = None,
) -> SimMaterial:
    return SimMaterial(
        color=color,
        emissionColor=emissionColor if emissionColor is not None else [0, 0, 0],
        specular=specular,
        shininess=shininess,
        reflectance=reflectance,
        texture=texture,
    )