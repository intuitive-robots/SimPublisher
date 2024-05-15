import io
import math
from pathlib import Path
from typing import Optional
from simpub.udata import UMesh, USubMesh, UTransform
import numpy as np  
import trimesh


def mat2transform(matrix):
  u, *_ = np.linalg.svd(matrix[:3, :3])
  pitch = np.arctan2(-u[2, 0], np.sqrt(u[2, 1]**2 + u[2, 2]**2))
  roll = np.arctan2(u[2, 1], u[2, 2])
  yaw = np.arctan2(-u[1, 0], u[0, 0])
  pos = matrix[:3, 3]
  return np.array([roll, pitch, yaw]), pos 

class MeshLoader:

  def fromFile(file : str | Path, mesh_type : Optional[str] = None) -> UMesh: 
    path = Path(file) if isinstance(file, str) else file
    return MeshLoader.fromBytes(path.read_bytes(), mesh_type or path.suffix[1:])
       

  def fromString(content : str, mesh_type : str) -> UMesh:
    return MeshLoader.fromBytes(content.encode(), mesh_type)
    

  def fromBytes(content : bytes, mesh_type : str) -> UMesh:

    with io.BytesIO(content) as data:
      mesh : trimesh.Trimesh = trimesh.load(data, file_type=mesh_type, texture=True)

    indices = mesh.faces.astype(np.int32)
    vertices = mesh.vertices.astype(np.float32)
    normals =  mesh.vertex_normals.astype(np.float32)
    uvs = mesh.visual.uv.astype(np.float32) if hasattr(mesh.visual, "uv") else None

      
    submesh, data = MeshLoader._build_mesh(indices, vertices, normals, uvs)
    
    return UMesh(
      tag=None,
      _data=data,
      submeshes=[submesh]
    )

  def _build_mesh(indices, vertices, norms, tex_coords) -> USubMesh:
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

    umesh = USubMesh(
      name="geometry_0",
      material=None,
      transform= UTransform(rotation=np.array([-math.pi / 2, 0, math.pi / 2])),
      indicesLayout=indices_layout, 
      verticesLayout=vertices_layout, 
      normalsLayout=normal_layout,
      uvLayout=uv_layout,
    )
    
    return umesh, bin_data
  