from __future__ import annotations

import io
from enum import Enum
from hashlib import md5
from typing import Literal, List, Optional, Tuple, TypedDict

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

class SimSceneConfig(TypedDict):
    name: str
    pos: List[float]
    rot: List[float]
    scale: List[float]

class LightConfig(TypedDict):
    name: str
    parent: Optional[str]
    lightType: Literal["Directional", "Point", "Spot"]
    color: List[float]        # [R, G, B]
    intensity: float          # Multiplier or Lux
    position: List[float]     # [X, Y, Z] in Unity coordinates
    direction: List[float]    # [X, Y, Z] forward vector for Unity
    range: float              # Attenuation distance
    spotAngle: float         # Full cone angle in degrees
    shadowType: str          # "None", "Hard", "Soft"

class TreeNode:
    def __init__(self) -> None:
        self.data: Optional[SimObject] = None
        self.children: List[TreeNode] = []

    def add_data(self, data: SimObject) -> None:
        self.data = data

    def add_child(self, child_node: TreeNode) -> None:
        self.children.append(child_node)

class SimScene:
    def __init__(self, config: SimSceneConfig) -> None:
        self.config: SimSceneConfig = config
        self.root: TreeNode = TreeNode()
        self.lights: List[LightConfig] = []

    @property
    def name(self) -> str:
        return self.config["name"]


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
    uv_data = None
    if hasattr(trimesh_obj, 'visual') and getattr(trimesh_obj.visual, 'uv', None) is not None:
        # Ensure float32 for Unity compatibility
        uv_data = getattr(trimesh_obj.visual, 'uv')
    return SimMesh(
        vertices=generate_bytes(vertices),
        indices=generate_bytes(indices),
        normals=generate_bytes(normals),
        uv=generate_bytes(uv_data) if uv_data is not None else None,
    )

def create_texture(
    image_flaten_array: np.ndarray,
    image_height: int,
    image_width: int,
    texture_scale: Tuple[int, int] = (1, 1),
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