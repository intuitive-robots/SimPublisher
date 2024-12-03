from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random
import json
import trimesh
import io
from hashlib import md5
import cv2


class VisualType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    CYLINDER = "CYLINDER"
    PLANE = "PLANE"
    QUAD = "QUAD"
    MESH = "MESH"
    NONE = "NONE"


@dataclass
class SimData:
    pass


@dataclass
class SimMaterial(SimData):
    # All the color values are within the 0 - 1.0 range
    color: List[float]
    emissionColor: Optional[List[float]] = None
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    texture: Optional[SimTexture] = None


@dataclass
class SimAsset(SimData):
    hash: str

    def update_raw_data(
        self,
        raw_data: Dict[str, bytes],
        new_data: bytes
    ) -> None:
        if self.hash in raw_data:
            raw_data.pop(self.hash)
        self.hash = md5(new_data).hexdigest()
        raw_data[self.hash] = new_data


@dataclass
class SimMesh(SimAsset):
    # (offset: bytes, count: int)
    verticesLayout: Tuple[int, int]
    indicesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]
    normalsLayout: Optional[Tuple[int, int]] = None

    def generate_normals(self, raw_data: Dict[str, bytes]) -> None:
        if self.normalsLayout is not None:
            return
        if self.hash not in raw_data:
            raise ValueError(f"Mesh data {self.hash} not found in raw data")
        bin_data = raw_data[self.hash]
        vert_offset, vert_size = self.verticesLayout
        vertices = np.frombuffer(
            bin_data[vert_offset:vert_offset + vert_size * 4],
            dtype=np.float32,
        ).reshape(-1, 3)
        indices_offset, indices_size = self.indicesLayout
        indices = np.frombuffer(
            bin_data[indices_offset:indices_offset + indices_size * 4],
            dtype=np.int32,
        ).reshape(-1, 3)
        trimesh_obj = trimesh.Trimesh(
            vertices=vertices,
            faces=indices,
            process=False,
        )
        # fix normals
        trimesh_obj.fix_normals()
        # save normals
        bin_buffer = io.BytesIO(raw_data[self.hash])
        # Normals
        norms = trimesh_obj.vertex_normals.astype(np.float32)
        norms = norms.flatten()
        self.normalsLayout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms)
        new_bin_data = bin_buffer.getvalue()
        # update hash
        self.update_raw_data(raw_data, new_bin_data)


@dataclass
class SimTexture(SimAsset):
    width: int = 0
    height: int = 0
    # TODO: add new texture type
    textureType: str = "2D"
    textureScale: Tuple[int, int] = field(default_factory=lambda: (1, 1))

    def compress(
        self,
        raw_data: Dict[str, bytes],
        max_texture_size_kb: int = 5000,
        min_scale: float = 0.5,
    ) -> None:
        if self.hash not in raw_data:
            raise ValueError(f"Texture data {self.hash} not found in raw data")
        bin_data = raw_data[self.hash]
        # compress the texture data
        max_texture_size = max_texture_size_kb * 1024
        # Compute scale factor based on texture size
        scale = np.sqrt(len(bin_data) / max_texture_size)
        # Adjust scale factor for small textures
        if scale < 1:  # Texture is already under the limit
            scale = 1  # No resizing needed
        elif scale < 1 + min_scale:  # Gradual scaling for small images
            scale = 1 + (scale - 1) * min_scale
        else:  # Normal scaling for larger textures
            scale = int(scale) + 1
        new_width = int(self.width // scale)
        new_height = int(self.height // scale)
        # Reshape and resize the texture data
        image_data = np.frombuffer(
            bin_data, dtype=np.uint8
        )
        image_data = cv2.resize(
            image_data.reshape(self.width, self.height, -1),
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
            # interpolation=cv2.INTER_AREA if scale > 2 else cv2.INTER_LINEAR,
        )
        self.width, self.height = new_height, new_height
        bin_data = image_data.astype(np.uint8).tobytes()
        self.update_raw_data(raw_data, bin_data)


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
        self.root: Optional[SimObject] = None
        self.id: str = str(random.randint(int(1e9), int(1e10 - 1)))
        self.raw_data: Dict[str, bytes] = dict()

    def to_string(self) -> str:
        if self.root is None:
            raise ValueError("Root object is not set")
        dict_data = {
            "root": asdict(self.root),
            "id": self.id,
        }
        return json.dumps(dict_data)

    def process_sim_obj(self, sim_obj: SimObject) -> None:
        for visual in sim_obj.visuals:
            if visual.mesh is not None:
                visual.mesh.generate_normals(self.raw_data)
        for child in sim_obj.children:
            self.process_sim_obj(child)
