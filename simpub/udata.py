from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

class UJointType(str, Enum):
  FIXED = "FIXED"
  FREE = "FREE"
  SLIDE = "SLIDE"
  BALL = "BALL"
  HINGE = "HINGE"

class UVisualType(str, Enum):
  MESH = "MESH",
  BOX = "BOX",
  CYLINDER = "CYLINDER",
  CAPSULE = "CAPSULE",
  PLANE = "PLANE",
  SPHERE = "SPHERE",
  NONE = "NONE"

class UAssetType(str, Enum):
  MESH = "MESH",
  MATERIAL = "MATERIAL",
  TEXTURE = "TEXTURE"


@dataclass
class UTransform:
  position: np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  rotation : np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  scale : np.ndarray = field(default_factory=lambda: np.ones((3, ), dtype=np.float32))


@dataclass
class UAsset:
  tag : str
  type : UAssetType = field(init=False)


@dataclass
class USubMesh:
  # Its important to keep indices as np.int32 and normals and vertices as np.float32
  name : str
  material : Optional[str]
  transform : UTransform
  indicesLayout : tuple[int, int] # (offset : bytes, count : int)
  normalsLayout : tuple[int, int] # (offset : bytes, count : int)
  verticesLayout : tuple[int, int] # (offset : bytes, count : int)
  uvLayout : tuple[int, int]

@dataclass
class UMesh(UAsset):
  submeshes : list[USubMesh]
  _data : bytes
  type = UAssetType.MESH
  hash : str = ""

@dataclass
class UMaterial(UAsset): # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
  color : np.ndarray[np.float32]  # All the color values are within the 0 - 1.0 range 
  emissionColor : np.ndarray[np.float32] 
  specular : float = 0.5
  shininess : float = 0.5
  reflectance : float = 0
  texture : Optional[str] = None
  type = UAssetType.MATERIAL

@dataclass
class UTexture(UAsset): # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
  width : int = 0
  height : int = 0
  mark : str = "none"
  builtin : str = "none"
  textype : str = "cube"
  gridlayout : str = "....."
  hflip : bool = False
  vflip : bool = False
  type = UAssetType.TEXTURE
  rgb1 : np.ndarray[np.float32] = field(default_factory=lambda:np.array([0.8, 0.8, 0.8]))
  rgb2 : np.ndarray[np.float32] = field(default_factory=lambda:np.array([0.5, 0.5, 0.5]))
  markrgb : np.ndarray[np.int32] = field(default_factory=lambda: np.array([0, 0, 0]))
  gridsize : np.ndarray[np.int32] = field(default_factory=lambda: np.array([1, 1]))
  hash : str = ""
  _data : bytes | None = None
  

"""
Scene data
"""
@dataclass
class UVisual:
  name : str
  type : str
  asset : str
  material : str
  transform : UTransform
  color : list[float]


@dataclass 
class UJoint:  
  name : str
  transform : UTransform
  link : "UBody"

  type : UJointType = UJointType.FIXED
  maxrot : float = 0.0
  minrot : float = 0.0
  axis : list[float] = field(default_factory=lambda: np.array([0, 0, 0]))


@dataclass
class UBody:
  name : str
  visuals : list[UVisual]
  joints : list[UJoint]

  def get_joints(self, include : set[UJointType] = {type for type in UJointType}, exclude : set[UJointType] = {UJointType.FIXED}) -> list[UJoint]:
    joints = [joint for joint in self.joints if joint.type not in exclude and joint.type in include]
    found = [joint.link for joint in self.joints]
    while found and (current := found.pop()):
      connected_joints = current.joints
      joints += [joint for joint in connected_joints if joint.type not in exclude and joint.type in include]
      found += [joint.link for joint in connected_joints]
    return joints


@dataclass
class UScene:
  id : str
  assets : dict[str, UAsset]
  objects : list[UBody]