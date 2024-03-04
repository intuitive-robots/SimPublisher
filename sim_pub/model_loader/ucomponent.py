from dataclasses import dataclass, asdict
from unicodedata import name
from xml.etree.ElementTree import Element as XMLNode
from enum import Enum
import abc
from typing import List, Optional, Tuple, TypeVar

from .utils import get_name, singleton


class UHeaderType(str, Enum):
	ENTITY = "ENTITY"
	MESH = "MESH"
	SHAPE = "SHAPE"
	UPDATE = "UPDATE"
	BEACON = "BEACON"
	SPAWN  = "SPAWN"
	DATA = "DATA"


class UJointType(str, Enum):
	REVOLUTE  = "REVOLUTE"
	PRISMATIC = "PRISMATIC"
	SPHERICAL = "SPHERIACAL"
	PLANAR    = "PLANAR"
	FIXED     = "FIXED"


class UVisualType(str, Enum):
	GEOMETRIC = "GEOMETRIC"
	BOX       = "BOX"
	SPHERE    = "SPHERE"
	CYLINDER  = "CYLINDER"
	CAPSULE   = "CAPSULE"
	PLANE     = "PLANE"
	MESH      = "MESH"

class UMetaEntity(abc.ABC):
	def __init__(self) -> None:
		super().__init__()

	@abc.abstractmethod
	def to_dict(self) -> dict:
		pass

@dataclass
class UComponent(UMetaEntity):
	name: str
	def to_dict(self) -> dict:
		return asdict(self)

class UMaterial(UComponent):
	specular : List[float]
	diffuse : List[float]
	ambient : List[float]
	glossiness : float


class UMesh(UComponent):
  name : str
  pos : List[float]
  rot : List[float]
  scale : List[float]
  indices : List[int]
  vertices : List[List[float]]
  normals : List[List[float]]
  material : UMaterial = None


class UVisual(UComponent):
	type : UVisualType
	pos : List[float]
	rot : List[float] = [0.0, 0.0, 0.0]
	scale : List[float]
	meshes : List[UMesh] = []
	
	def to_dict(self) -> dict:
		return {
			"name": self.name,
			"type": self.type,
			"pos": self.pos,
			"rot": self.rot,
			"scale" : self.scale,
			"meshes" : self.meshes,
		}

class UGameObjectBase(UMetaEntity):

	def __init__(self, name) -> None:
		super().__init__()
		self.name = name
		self.pos: list[float] = [0, 0, 0]
		self.rot: list[float] = [0, 0, 0]
		self.moveable: bool = True
		self.parent: UGameObjectBase

	def to_dict(self) -> dict:
		return {
			"name": self.name,
			"pos": self.pos,
			"rot": self.rot,
			"moveable": self.moveable,
			"parent": self.parent.name,
		}

class UGameObject(UGameObjectBase):
	def __init__(
		self,
		name,
		parent: UGameObjectBase,
	) -> None:
		super().__init__(name)
		self.parent = parent
		self.visual: List[UVisual] = list()
		self.children: List[UGameObjectBase] = list()

	def add_child(self, child: UGameObjectBase) -> None:
		self.children.append(child)
		child.parent = self

	def add_visual(self, visual: UVisual):
		self.visual.append(visual)

	def to_dict(self) -> dict:
		return {
			**super().to_dict(),
			"children": [child.name for child in self.children],
			"visual": [item.to_dict() for item in self.visual]
		}
  
class UVisualGameObject(UGameObject):
	def __init__(self, name, parent: UGameObject) -> None:
		super().__init__(name, parent)
		self.visual: UVisual

@singleton
class EmptyGameObject(UGameObject):
	pass

@singleton
class SceneRoot(UGameObject):
	def __init__(self) -> None:
		super().__init__("SceneRoot", parent=None)


class UJoint(UMetaEntity):
	def __init__(self) -> None:
		super().__init__()
		self.name: str
		self.type: str
		self.parent: str
		self.child: str
		self.parent: str
		self.child: str
		self.axis: List[float]
		self.jointRange: List[float]
