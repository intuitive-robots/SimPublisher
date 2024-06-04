from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
import random


class SimJointType(str, Enum):
  FIXED = "FIXED"
  FREE = "FREE"
  SLIDE = "SLIDE"
  BALL = "BALL"
  HINGE = "HINGE"

class SimVisualType(str, Enum):
  CYLINDER = "CYLINDER",
  CAPSULE = "CAPSULE",
  SPHERE = "SPHERE",
  PLANE = "PLANE",
  MESH = "MESH",
  NONE = "NONE",
  BOX = "BOX",

class SimAssetType(str, Enum):
  MATERIAL = "MATERIAL",
  TEXTURE = "TEXTURE",
  MESH = "MESH",

@dataclass
class SimAsset:
  tag : str

  def __repr__(self):
    return f"<{type(self)} tag={self.tag}>"

@dataclass
class SimMesh(SimAsset):
  dataID : str 
  # (offset : bytes, count : int)
  indicesLayout   : Tuple[int, int]
  normalsLayout   : Tuple[int, int]
  verticesLayout  : Tuple[int, int]
  uvLayout        : Tuple[int, int]
  type : SimAssetType = SimAssetType.MESH

@dataclass
class SimMaterial(SimAsset): 
  # All the color values are within the 0 - 1.0 range 
  color : np.ndarray
  emissionColor : np.ndarray
  specular : float = 0.5
  shininess : float = 0.5
  reflectance : float = 0
  texture : Optional[str] = None
  texsize : Tuple [int, int] = (1, 1)
  type : SimAssetType = SimAssetType.MATERIAL

@dataclass
class SimTexture(SimAsset):
  dataID : str
  width : int = 0
  height : int = 0
  textype : str = "cube"
  type : SimAssetType = SimAssetType.TEXTURE
  

"""
Scene data
"""
@dataclass
class SimTransform:
  position: np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  rotation : np.ndarray = field(default_factory=lambda: np.zeros((3, ), dtype=np.float32))
  scale : np.ndarray = field(default_factory=lambda: np.ones((3, ), dtype=np.float32))

  def __add__(self, other : "SimTransform"):
    return SimTransform(
      position=self.position + other.position,
      rotation=self.rotation + other.rotation,
      scale=self.scale * other.scale
    )
  
  def __repr__(self) -> str:
    return f"<SimTransform pos={self.position.tolist()} rot={self.rotation.tolist()} scale={self.scale.tolist()}>"


@dataclass
class SimVisual:
  type : str
  mesh : str
  material : str
  transform : SimTransform
  color : List[float]

  def __repr__(self) -> str:
    return f"<SimVisual tupe={self.type}>"


@dataclass 
class SimJoint:  
  name : str
  initial : float = 0.0
  maxrot : float = 0.0
  minrot : float = 0.0
  transform : SimTransform = field(default_factory=SimTransform)
  type : SimJointType = SimJointType.FIXED
  axis : List[float] = field(default_factory=lambda: np.array([0, 0, 0]))

  
  def __repr__(self) -> str:
    return f"<SimJoint name={self.name} type={self.type}>"

@dataclass
class SimBody:
  name : str
  visuals : List[SimVisual]
  joints : List[SimJoint]
  bodies : List["SimBody"]

  def get_joints(self, *types) -> List[SimJoint]:
    select: set = set(types) if len(types) > 0 else set(SimJointType)
    joints = list()
    found = [self]
    while found and (current := found.pop()):
      connected_joints = current.joints
      joints.extend(joint for joint in connected_joints if joint.type in select)
      found.extend(current.bodies)
    return joints

  def get_parent_body(self, joint : SimJoint):
    found = [self]
    while found and (current := found.pop()):
      if joint in current.joints:
        return current
      found.extend(current.bodies)
    return None
  
  def __repr__(self) -> str:
    return f"<SimBody visuals={len(self.visuals)} bodies={len(self.bodies)} joints={len(self.joints)}>"

  
@dataclass
class SimScene:
  worldbody : SimBody = None
  id : str = str(random.randint(int(1e9), int(1e10 - 1)))
  meshes : List[SimMesh] = field(default_factory=list)
  textures : List[SimMesh] = field(default_factory=list)
  materials : List[SimMesh] = field(default_factory=list)
  _meta_data : Dict[str, Any] = field(default_factory=dict)
  _raw_data : Dict[str, bytes] = field(default_factory=dict)