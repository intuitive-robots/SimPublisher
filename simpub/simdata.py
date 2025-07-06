from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random
import json
import trimesh
import io
from hashlib import md5
import cv2
import abc


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

    def to_dict(self):
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.metadata.get("exclude", False)
        }


@dataclass
class SimMaterial(SimData):
    # All the color values are within the 0 - 1.0 range
    color: List[float]
    emissionColor: Optional[List[float]] = None
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    texture: Optional[SimTexture] = None

    def to_dict(self):
        data = super().to_dict()
        if self.texture is not None:
            data["texture"] = self.texture.to_dict()
        return data


@dataclass
class SimAsset(SimData):
    hash: str
    # scene: SimScene = field(init=True, metadata={"exclude": True})

    # def __post_init__(self):
    #     raw_data = self.generate_raw_data()
    #     self.hash = self.generate_hash(raw_data)

    @abc.abstractmethod
    def generate_raw_data(self) -> bytes:
        raise NotImplementedError

    @staticmethod
    def generate_hash(data: bytes) -> str:
        return md5(data).hexdigest()

    @staticmethod
    def write_to_buffer(
        bin_buffer: io.BytesIO,
        data: np.ndarray,
    ) -> Tuple[int, int]:
        # change all float nparray to float32 and all int nparray to int32
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        elif data.dtype == np.int64:
            data = data.astype(np.int32)
        byte_data = data.tobytes()
        layout = bin_buffer.tell(), len(byte_data)
        bin_buffer.write(byte_data)
        return layout

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
    normalsLayout: Tuple[int, int]
    uvLayout: Optional[Tuple[int, int]] = None

    @staticmethod
    def create_mesh(
        scene: SimScene,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: Optional[np.ndarray] = None,
        face_normals: Optional[np.ndarray] = None,
        mesh_texcoord: Optional[np.ndarray] = None,
        vertex_uvs: Optional[np.ndarray] = None,
        faces_uv: Optional[np.ndarray] = None,
    ) -> SimMesh:
        uvs = None
        if vertex_uvs is not None:
            assert vertex_uvs.shape[0] == vertices.shape[0], (
                f"Number of vertex uvs ({vertex_uvs.shape[0]}) must be equal "
                f"to number of vertices ({vertices.shape[0]})"
            )
            uvs = vertex_uvs.flatten()
        elif faces_uv is not None and mesh_texcoord is None:
            vertices, faces = SimMesh.generate_vertex_by_faces(vertices, faces)
            uvs = faces_uv.flatten()
        elif mesh_texcoord is not None and faces_uv is not None:
            if mesh_texcoord.shape[0] == vertices.shape[0]:
                uvs = mesh_texcoord
            else:
                vertices, faces, uvs, vertex_normals = (
                    SimMesh.generate_vertex_uv_from_face_uv(
                        vertices,
                        faces_uv,
                        faces,
                        mesh_texcoord,
                    )
                )
            assert uvs.shape[0] == vertices.shape[0], (
                f"Number of mesh texcoords ({mesh_texcoord.shape[0]}) must be "
                f"equal to number of vertices ({vertices.shape[0]})"
            )
            uvs = uvs.flatten()
        # create trimesh object
        trimesh_obj = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            process=False,
        )
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
        assert indices.size % 3 == 0, (
            f"Number of indices ({indices.size}) must be a multiple of 3"
        )
        # write to buffer
        bin_buffer = io.BytesIO()
        vertices_layout = SimMesh.write_to_buffer(bin_buffer, vertices)
        indices_layout = SimMesh.write_to_buffer(bin_buffer, indices)
        normals_layout = SimMesh.write_to_buffer(bin_buffer, normals)
        # UVs
        uv_layout = (0, 0)
        if uvs is not None:
            assert uvs.size == num_vertices * 2, (
                f"Number of vertex uvs ({uvs.shape[0]}) must be equal to "
                f"number of vertices ({num_vertices})"
            )
            uv_layout = SimMesh.write_to_buffer(bin_buffer, uvs)
        bin_data = bin_buffer.getvalue()
        hash = SimMesh.generate_hash(bin_data)
        scene.raw_data[hash] = bin_data
        return SimMesh(
            hash=hash,
            verticesLayout=vertices_layout,
            indicesLayout=indices_layout,
            normalsLayout=normals_layout,
            uvLayout=uv_layout,
        )

    @staticmethod
    def generate_vertex_by_faces(
        vertices: np.ndarray,
        faces: np.ndarray,
    ):
        new_vertices = []
        for face in faces:
            new_vertices.append(vertices[face])
        new_vertices = np.concatenate(new_vertices)
        new_faces = np.arange(new_vertices.shape[0]).reshape(-1, 3)
        return new_vertices, new_faces

    @staticmethod
    def generate_vertex_uv_from_face_uv(
        vertices: np.ndarray,
        face_texcoord: np.ndarray,
        faces: np.ndarray,
        mesh_texcoord: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        new_vertices = []
        new_faces = []
        vertex_texcoord = []
        for face, face_texcoord in zip(faces, face_texcoord):
            new_vertices.append(vertices[face])
            vertex_texcoord.append(mesh_texcoord[face_texcoord])
        for item in new_vertices:
            assert item.shape == (3, 3), f"Shape of item is {item.shape}"
        new_vertices = np.concatenate(new_vertices)
        vertex_texcoord = np.concatenate(vertex_texcoord)
        new_faces = np.arange(new_vertices.shape[0]).reshape(-1, 3)
        return new_vertices, new_faces, vertex_texcoord, None


@dataclass
class SimTexture(SimAsset):
    width: int
    height: int
    # TODO: new texture type
    textureType: str
    textureScale: Tuple[int, int]

    @staticmethod
    def compress_image(
        image: np.ndarray,
        height: int,
        width: int,
        max_texture_size_kb: int = 5000,
        min_scale: float = 0.5,
    ) -> Tuple[np.ndarray, int, int]:
        # compress the texture data
        max_texture_size = max_texture_size_kb * 1024
        # Compute scale factor based on texture size
        scale = np.sqrt(image.nbytes / max_texture_size)
        # Adjust scale factor for small textures
        if scale < 1:  # Texture is already under the limit
            scale = 1  # No resizing needed
        elif scale < 1 + min_scale:  # Gradual scaling for small images
            scale = 1 + (scale - 1) * min_scale
        else:  # Normal scaling for larger textures
            scale = int(scale) + 1
        new_width = int(width // scale)
        new_height = int(height // scale)
        # Reshape and resize the texture data
        compressed_image = cv2.resize(
            image.reshape(height, width, -1),
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
            # interpolation=cv2.INTER_AREA if scale > 2 else cv2.INTER_LINEAR,
        )
        return compressed_image, new_height, new_width

    @staticmethod
    def create_texture(
        image: np.ndarray,
        height: int,
        width: int,
        scene: SimScene,
        texture_scale: Tuple[int, int] = field(default_factory=lambda: (1, 1)),
        texture_type: str = "2D",
    ) -> SimTexture:
        image = image.astype(np.uint8)
        image, height, width = SimTexture.compress_image(image, height, width)
        image_byte = image.astype(np.uint8).tobytes()
        hash = SimTexture.generate_hash(image_byte)
        scene.raw_data[hash] = image_byte
        return SimTexture(
            hash=hash,
            width=image.shape[1],
            height=image.shape[0],
            textureType=texture_type,
            textureScale=texture_scale,
        )


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
    mesh: Optional[SimMesh] = None
    material: Optional[SimMaterial] = None
    # TODO： easily set up transparency
    # def setup_transparency(self):
    #     if self.material is not None:
    #         self.material = self

    def to_string(self) -> str:
        dict_data = {
            "name": self.name,
            "type": self.type.value,
            "trans": self.trans.to_dict(),
            "mesh": self.mesh.to_dict() if self.mesh else None,
            "material": self.material.to_dict() if self.material else None,
        }
        return json.dumps(dict_data)


@dataclass
class SimObject(SimData):
    name: str
    trans: SimTransform = field(default_factory=SimTransform)
    visuals: List[SimVisual] = field(default_factory=list)
    children: List[SimObject] = field(default_factory=list)

    def to_string(self, sim_scene: SimScene, parent: Optional[SimObject]) -> str:
        # visuals_data = [visual.to_dict() for visual in self.visuals]
        # children_data = [child.to_dict() for child in self.children]
        dict_data = {
            "name": self.name,
            "parentName": parent.name if parent else "",
            "sceneName": sim_scene.name,
            "trans": self.trans.to_dict(),
        }
        return json.dumps(dict_data)


class SimScene:
    def __init__(self) -> None:
        self.root: Optional[SimObject] = None
        self.name: str = "DefaultSceneName"
        self.raw_data: Dict[str, bytes] = dict()

    def to_string(self) -> str:
        if self.root is None:
            raise ValueError("Root object is not set")
        dict_data = {
            "name": self.name,
        }
        return json.dumps(dict_data)

    def process_sim_obj(self, sim_obj: SimObject) -> None:
        # for visual in sim_obj.visuals:
        #     if visual.mesh is not None:
        #         visual.mesh.generate_normals(self.raw_data)
        for child in sim_obj.children:
            self.process_sim_obj(child)
