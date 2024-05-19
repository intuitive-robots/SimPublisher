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
class UAsset:
  tag : str
  type : UAssetType = field(init=False)

@dataclass
class UMesh(UAsset):
  _data : bytes
  indicesLayout : tuple[int, int] # (offset : bytes, count : int)
  normalsLayout : tuple[int, int] # (offset : bytes, count : int)
  verticesLayout : tuple[int, int] # (offset : bytes, count : int)
  uvLayout : tuple[int, int]
  hash : str = ""
  type = UAssetType.MESH

@dataclass
class UMaterial(UAsset): # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
  color : np.ndarray[np.float32]  # All the color values are within the 0 - 1.0 range 
  emissionColor : np.ndarray[np.float32] 
  specular : float = 0.5
  shininess : float = 0.5
  reflectance : float = 0
  texture : Optional[str] = None
  texsize : tuple [int, int] = (1, 1)
  type = UAssetType.MATERIAL

@dataclass
class UTexture(UAsset): # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
  width : int = 0
  height : int = 0
  textype : str = "cube"
  type = UAssetType.TEXTURE
  hash : str = ""
  _data : bytes | None = None
  

"""
Scene data
"""
@dataclass
class UTransform:
  position: np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  rotation : np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  scale : np.ndarray = field(default_factory=lambda: np.ones((3, ), dtype=np.float32))


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
  body : "UBody"

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
    found = [joint.body for joint in self.joints]
    while found and (current := found.pop()):
      connected_joints = current.joints
      joints += [joint for joint in connected_joints if joint.type not in exclude and joint.type in include]
      found += [joint.body for joint in connected_joints]
    return joints


@dataclass
class UScene:
  id : str
  assets : dict[str, UAsset]
  worldbody : UBody