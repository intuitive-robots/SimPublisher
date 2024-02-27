from dataclasses import dataclass
from xml.etree.ElementTree import Element as XMLNode
from enum import Enum
import abc
from typing import List, Optional, Tuple, TypeVar

from .utils import singleton


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
	def to_string(self) -> str:
		pass

@dataclass
class UComponent(UMetaEntity):
	def to_string(self) -> str:
		return super().to_string()

class UMaterial(UComponent):

	specular : List[float]
	diffuse : List[float]
	ambient : List[float]
	glossiness : float

	def to_string(self) -> str:
		return super().to_string()

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
	rot : List[float]
	scale : List[float]
	meshes : List[UMesh]

class UGameObjectBase(UMetaEntity):

	def __init__(self) -> None:
		super().__init__()
		self.pos: list[float] = [0, 0, 0]
		self.rot: list[float] = [0, 0, 0]
		self.static: bool = False
		self.parent: UGameObjectBase

	def to_string(self) -> str:
		return super().to_string()

class UGameObject(UGameObjectBase):
	def __init__(
		self,
		parent: UGameObjectBase,
	) -> None:
		super().__init__()
		self.parent = parent
		self.visual_child: UGameObject = None
		self.children: List[UGameObjectBase] = list()

	def add_child(self, child: UGameObjectBase) -> None:
		self.children.append(child)

	def add_visual(self, visual: UVisual):
		if self.visual_child is None:
			self.visual_child = UGameObject(parent=self)
		visual_object = UVisualGameObject(self.visual_child)
		visual_object.visual = visual
		self.visual_child.add_child(visual_object)


class UVisualGameObject(UGameObject):
	def __init__(self, parent: UGameObject) -> None:
		super().__init__(parent)
		self.visual: UVisual

@singleton
class EmptyGameObject(UGameObject):
	pass

@singleton
class SceneRoot(UGameObject):
	def __init__(self) -> None:
		super().__init__(parent=None)


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
