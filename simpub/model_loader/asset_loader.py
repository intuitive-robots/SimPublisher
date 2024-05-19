import io
import math
from pathlib import Path
from typing import Optional

import simpub
from simpub.simdata import SimMesh, SimTexture
import numpy as np  
import trimesh

from hashlib import md5
from PIL import Image


class TextureLoader:
  RES_PATH = Path(simpub.__file__).parent

  @staticmethod
  def fromBuiltin(name : str, builtin_name : str, tint : Optional[np.ndarray] = None):
    img : Image.Image
    match builtin_name:
      case "checker":
        img = Image.open(TextureLoader.RES_PATH / "res/image/builtin/checker_grey.png").convert("RGBA")
      case "gradient":
        img = Image.new("RGBA", (32, 32))
    
    if tint is not None: img = TextureLoader.tint(img, tint)

    width, height = img.size 
    tex_data = img.tobytes()
    texture_hash = md5(tex_data).hexdigest()

    return SimTexture(
      tag = name,
      width = width,
      height= height,
      textype="2d",
      hash = texture_hash,
      _data = tex_data
    )

  @staticmethod
  def fromBytes(name : str, content : bytes, texture_type : str, tint : Optional[np.ndarray] = None):
    with io.BytesIO(content) as file_data:
      img : Image.Image = Image.open(file_data).convert("RGBA")
    
    
    if tint is not None: img = TextureLoader.tint(img, tint)

    width, height = img.size 
    tex_data = img.tobytes()
    texture_hash = md5(tex_data).hexdigest()

    return SimTexture(
      tag = name,
      width = width,
      height= height,
      textype=texture_type,
      hash = texture_hash,
      _data = tex_data
    )
  
  @staticmethod
  def tint(img : Image.Image, tint : np.ndarray):
    r, g, b, a = img.split()

    # Apply the tint to each band
    r = r.point(lambda i: int(i * tint[0]))
    g = g.point(lambda i: int(i * tint[1]))
    b = b.point(lambda i: int(i * tint[2]))

    # Merge the bands back together
    return Image.merge('RGBA', (r, g, b, a))


class MeshLoader:
  @staticmethod
  def fromFile(file : str | Path, mesh_type : Optional[str] = None, name : Optional[str] = None, **kwargs) -> SimMesh: 
    path = Path(file) if isinstance(file, str) else file
    return MeshLoader.fromBytes(name or file.name, path.read_bytes(), mesh_type or path.suffix[1:], **kwargs)
       

  @staticmethod
  def fromString(name : str, content : str, mesh_type : str, **kwargs) -> SimMesh:
    return MeshLoader.fromBytes(content.encode(), mesh_type, **kwargs)
    

  @staticmethod
  def fromBytes(name : str, content : bytes, mesh_type : str, **kwargs) -> SimMesh:
    
    with io.BytesIO(content) as data:
      mesh : trimesh.Trimesh = trimesh.load(data, file_type=mesh_type, texture=True)
      if kwargs.get("scale") is not None: mesh.apply_scale(kwargs["scale"])
      mesh.apply_transform(trimesh.transformations.euler_matrix(math.pi, math.pi / 2.0, -math.pi / 2.0))

    indices = mesh.faces.astype(np.int32)
    vertices = mesh.vertices.astype(np.float32)
    normals =  mesh.vertex_normals.astype(np.float32)
    uvs = mesh.visual.uv.astype(np.float32) if hasattr(mesh.visual, "uv") else None

      
    return MeshLoader._build_mesh(indices, vertices, normals, uvs, name, hash, **kwargs)
    

  @staticmethod
  def _build_mesh(indices, vertices, norms, tex_coords, name, hash, **kwargs) -> SimMesh:
    bin_data = bytes()

    ## Vertices
    verts = vertices.flatten()
    vertices_layout = len(bin_data), verts.shape[0]
    bin_data = bin_data + verts.tobytes()

    ## Normals
    norms = norms.flatten()
    normal_layout = len(bin_data), norms.shape[0]
    bin_data += norms.tobytes() 

    ## Indices
    indices = indices.flatten() 
    indices_layout = len(bin_data), indices.shape[0]
    bin_data += indices.tobytes() 

    ## Texture coords
    uv_layout = 0, 0
    if tex_coords is not None:
      tex_coords[:, 1] = 1 - tex_coords[:, 1]
      uvs = tex_coords.flatten() 
      uv_layout = len(bin_data), uvs.shape[0]
      bin_data += uvs.tobytes()

    hash = md5(bin_data).hexdigest()

    return SimMesh(
      tag=name,
      hash=hash,
      indicesLayout=indices_layout, 
      verticesLayout=vertices_layout, 
      normalsLayout=normal_layout,
      uvLayout=uv_layout,
      _data = bin_data
    )
    
  