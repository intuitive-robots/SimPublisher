import io
import math
from pathlib import Path
from typing import Optional, Tuple

import simpub
from simpub.simdata import SimMesh, SimTexture
import numpy as np
import trimesh

from hashlib import md5
from PIL import Image


class TextureLoader:
    RES_PATH = Path(simpub.__file__).parent

    @staticmethod
    def fromBuiltin(
        name: str,
        builtin_name: str,
        tint: Optional[np.ndarray] = None
    ) -> Tuple[SimTexture, bytes]:
        img: Image.Image
        if builtin_name == "checker":
            img = Image.open(
                TextureLoader.RES_PATH / "parser/mjcf/builtin/checker_grey.png"
            ).convert("RGBA")
        elif builtin_name in {"gradient", "flat"}:
            img = Image.new("RGBA", (256, 256), (1, 1, 1, 1))
        else:
            raise RuntimeError("Invalid texture builtin", builtin_name)

        if tint is not None:
            img = TextureLoader.tint(img, tint)

        width, height = img.size
        tex_data = img.tobytes()
        texture_hash = md5(tex_data).hexdigest()

        texture = SimTexture(
            id=name,
            width=width,
            height=height,
            textureType="2d",
            dataHash=texture_hash
        )
        return texture, tex_data

    @staticmethod
    def from_bytes(
        name: str,
        content: bytes,
        texture_type: str,
        tint: Optional[np.ndarray] = None
    ) -> Tuple[SimTexture, bytes]:
        with io.BytesIO(content) as file_data:
            img = Image.open(file_data).convert("RGBA")
        if tint is not None:
            img = TextureLoader.tint(img, tint)
        width, height = img.size
        tex_data = img.tobytes()
        texture_hash = md5(tex_data).hexdigest()

        texture = SimTexture(
            id=name,
            width=width,
            height=height,
            textureType=texture_type,
            dataHash=texture_hash,
        )

        return texture, tex_data

    @staticmethod
    def tint(img: Image.Image, tint: np.ndarray):
        r, g, b, a = img.split()

        # Apply the tint to each band
        r = r.point(lambda i: i * tint[0])
        g = g.point(lambda i: i * tint[1])
        b = b.point(lambda i: i * tint[2])

        # Merge the bands back together
        return Image.merge('RGBA', (r, g, b, a))


class MeshLoader:
    @staticmethod
    def from_file(
        file_path: str,
        name: Optional[str] = None,
        scale: Optional[np.ndarray] = None,
    ) -> Tuple[SimMesh, bytes]:
        mesh = trimesh.load_mesh(file_path)
        return MeshLoader.from_loaded_mesh(mesh, name, scale)

    @staticmethod
    def from_bytes(
        name: str,
        content: bytes,
        mesh_type: str,
        scale: np.ndarray
    ) -> Tuple[SimMesh, bytes]:
        with io.BytesIO(content) as data:
            mesh: trimesh.Trimesh = trimesh.load_mesh(
                data, file_type=mesh_type, texture=True
            )
        name = Path(name)
        return MeshLoader.from_loaded_mesh(mesh, name, scale)

    @staticmethod
    def from_loaded_mesh(
        mesh: trimesh.Trimesh,
        name: str,
        scale: Optional[np.ndarray] = None
    ) -> Tuple[SimMesh, bytes]:
        if scale is not None:
            mesh.apply_scale(scale)
        mesh = mesh.apply_transform(
            trimesh.transformations.euler_matrix(
                -math.pi / 2.0, math.pi / 2.0, 0
            )
        )
        indices = mesh.faces.astype(np.int32)
        bin_buffer = io.BytesIO()
        # Vertices
        verts = mesh.vertices.astype(np.float32)
        verts[:, 2] = -verts[:, 2]
        verts = verts.flatten()
        vertices_layout = bin_buffer.tell(), verts.shape[0]
        bin_buffer.write(verts)
        # Normals
        norms = mesh.vertex_normals.astype(np.float32)
        norms[:, 2] = -norms[:, 2]
        norms = norms.flatten()
        normal_layout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms) 
        # Indices
        indices = mesh.faces.astype(np.int32)
        indices = indices[:, [2, 1, 0]]
        indices = indices.flatten()
        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)
        # Texture coords
        uv_layout = (0, 0)
        if hasattr(mesh.visual, "uv"):
            uvs = mesh.visual.uv.astype(np.float32)
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs = uvs.flatten()
            uv_layout = bin_buffer.tell(), uvs.shape[0]

        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()

        mesh = SimMesh(
            id=name,
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            normalsLayout=normal_layout,
            uvLayout=uv_layout,
            dataHash=hash
        )
        return mesh, bin_data
