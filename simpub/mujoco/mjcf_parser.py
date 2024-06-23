from collections import defaultdict
import copy
from dataclasses import dataclass
import random
from typing import List, Optional, Set
import numpy as np
from simpub.loaders.asset_loader import MeshLoader, TextureLoader
from simpub.simdata import SimBody, SimJoint, SimJointType, SimMaterial, SimScene, SimTransform, SimVisual, SimVisualType
from xml.etree import ElementTree
from pathlib import Path
from xml.dom import minidom

from scipy.spatial.transform import Rotation

class MJConv:  
  
  VISUAL_TYPE = {
    "plane" : SimVisualType.PLANE, 
    "sphere" : SimVisualType.SPHERE, 
    "capsule" : SimVisualType.CAPSULE, 
    "ellipsoid" : SimVisualType.CAPSULE, 
    "cylinder" : SimVisualType.CYLINDER,
    "box" : SimVisualType.BOX,
    "mesh" : SimVisualType.MESH
  }  
  
  @staticmethod
  def to_quat(quat) -> np.ndarray:
    return np.array([quat[1], quat[3], quat[2], -quat[0]])

  @staticmethod
  def to_pos(pos) -> np.ndarray:  
    return np.array([pos[0], pos[2], pos[1]]) 

  @staticmethod
  def to_visual_type(visual_type : str) -> SimVisualType:
    if visual_type not in MJConv.VISUAL_TYPE: raise RuntimeError("Invalid visual type")
    return MJConv.VISUAL_TYPE[visual_type]

@dataclass
class MJCFScene(SimScene):
  _path : Path = None
  _xmlElement : ElementTree.Element = None
  _visual_group : Set[int] = None 
  
  @staticmethod
  def from_string(content : str, path : Path, visual_group = None):
    return MJCFScene(
      _xmlElement=ElementTree.fromstring(content), 
      _visual_group = visual_group or set(range(10)),
      _path=path,
    )
  
  @staticmethod
  def from_file(file : Path, visual_group = None):
    path : Path = file if isinstance(file, Path) else Path(file)
    return MJCFScene(
      _xmlElement=ElementTree.fromstring(path.read_text()),
      _visual_group = visual_group or set(range(10)),
      _path=path,
    )

  def __post_init__(self):

    if self._path is None or self._xmlElement is None:
      raise RuntimeError("MJCFScene has to instancated from .from_string or .from_file")

    self.id = str(random.randint(int(1e9), int(1e10 - 1)))
    self.worldbody = None

    self.merge_includes()
    self.load_compiler()
    self.load_defaults()
    self.load_assets()
    self.load_worldbody()

    self.xmlstr = minidom.parseString(ElementTree.tostring(self._xmlElement)).toprettyxml(indent="   ")

  def _string2list(self, data : str, dtype : np.dtype = np.float32) -> np.ndarray:
    return np.fromstring(data, dtype=dtype, sep=' ')

  def merge_includes(self):
    while len(includes := list(self._xmlElement.iter("include"))):
      for include in includes:
        file = (self._path.parent / include.attrib["file"]).absolute()
        
        if not file.exists(): raise RuntimeError(f"mjcf file {file} not found")

        parsed_include = list(ElementTree.fromstring(file.read_text())) # parse
        parent = { ch : pa for pa in self._xmlElement.iter() for ch in pa}[include] # get the parent 
        index = { child : i for i, child in enumerate(parent)}[include]
        parent.remove(include)
        if parent != self._xmlElement:
          # Within default or the worlbody we add the children in place
          for i in range(index, index + len(parsed_include)):
            parent.insert(i, parsed_include[i - index])
          continue

        # If were in the root dir we merge the children
        for child in parsed_include:
          if (it := parent.find(child.tag)) is not None and len(list(it)) > 0:
            it.extend(list(child))
          elif it is not None and len(list(it)) == 0:
            it.attrib.update(child.attrib)
          else:
            parent.insert(index, child)

    with open("dump.xml", "w") as fp:
      fp.write(ElementTree.tostring(self._xmlElement, encoding="unicode"))

  def load_defaults(self):
    self.default = defaultdict(dict)
    default_container = self._xmlElement.find("./default")
    if default_container is None: return
    self.load_default(default_container, defaultdict(dict))
    

  def load_default(self, element : ElementTree.Element, default_tree : defaultdict):
    # depth first algorithm 
    # load all the default values
    for child in element:
      if child.tag == "default":
        self.load_default(child, copy.deepcopy(default_tree))
        continue

      for key, value in child.attrib.items(): 
        default_tree[child.tag][key] = value
  
    # save the default values for this class
    for tag in default_tree.keys():
      self.default[element.get("class", "main")][tag] = default_tree[tag]


  def load_compiler(self):
    compiler = self._xmlElement.find("./compiler")
    compiler = compiler if compiler is not None else dict()

    self.angle = compiler.get("angle") or "degree"
    self.eulerseq = compiler.get("eulerseq") or "xyz"

    self.meshdir    = self._path.parent / (compiler.get("meshdir") or compiler.get("assetdir") or "") 
    self.texturedir = self._path.parent / (compiler.get("texturedir") or compiler.get("assetdir") or "") 


  def rotation_from_object(self, obj : ElementTree.Element) -> np.ndarray: 
    # https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations
    result : np.ndarray = np.array([1, 0, 0, 0]) # mjcf default x = w
    if quat := obj.get("quat"):
      result = self._string2list(quat)
    elif euler := obj.get("euler"):
      euler = self._string2list(euler)
      result = Rotation.from_euler(self.eulerseq, euler, degrees=self.angle == "degree").as_quat()
    elif axisangle := obj.get("axisangle"):
      raise NotImplementedError()
    elif xyaxes := obj.get("xyaxes"):
      raise NotImplementedError()
    elif zaxis := obj.get("zaxis"):
      raise NotImplementedError()

    assert len(result) == 4
    return MJConv.to_quat(result)
  
  
  
  def load_visual(self, visual : ElementTree.Element) -> Optional[SimVisual]:   
    if int(visual.get("group", '0')) not in self._visual_group: return None
    
    transform = SimTransform(
      rotation=self.rotation_from_object(visual), 
      position=MJConv.to_pos(self._string2list(visual.get("pos", "0 0 0"))), 
      scale=self._string2list(visual.get("size", "1 1 1"))
    )
  
    type = MJConv.to_visual_type(visual.get("type", "sphere"))
    if type == SimVisualType.PLANE:
      transform.scale = np.abs(np.array([(transform.scale[0] or 100.0) * 2, 0.001, (transform.scale[1] or 100)* 2]))
    elif type == SimVisualType.BOX:
      transform.scale = np.abs(MJConv.to_pos(transform.scale * 2))
    elif type == SimVisualType.SPHERE:
      transform.scale = np.abs(MJConv.to_pos(np.array([transform.scale[0] * 2] * 3)))
    elif type ==  SimVisualType.CYLINDER:
      if len(transform.scale) == 3:
        transform.scale = np.abs(np.array([transform.scale[0], transform.scale[1], transform.scale[0]])) 
      else:
        transform.scale = np.abs(np.array([transform.scale[0] * 2, transform.scale[1], transform.scale[0] * 2])) 
    elif type == SimVisualType.CAPSULE:
      if len(transform.scale) == 3:
        transform.scale = np.abs(np.array([transform.scale[0], transform.scale[1], transform.scale[0]])) 
      # TODO  https://github.com/google-deepmind/mujoco/blob/main/unity/Runtime/Tools/MjEngineTool.cs
      # transform.localPosition = Vector3.Lerp(toPoint, fromPoint, 0.5f);
      # transform.localRotation = Quaternion.FromToRotation(
      #     Vector3.up, (toPoint - fromPoint).normalized);
    color = self._string2list(visual.get("rgba", "1 1 1 1"))

    assert len(color) == 4, visual.get("name")
    assert len(transform.scale) == 3
    return SimVisual(
      type=type,
      transform=transform,
      mesh=visual.get("mesh"),
      material=visual.get("material"),
      color=color
    )

  def load_joint(self, joint : ElementTree.Element, parent_name : str):
    range = self._string2list(joint.get("range", "1000000 100000"))
    if self.angle != "degree": range = np.degrees(range)
    return SimJoint(
      name=joint.get("name", parent_name),
      axis=MJConv.to_pos(self._string2list(joint.get("axis", "1 0 0"))),
      minrot=float(range[0]),
      maxrot=float(range[1]),
      type=SimJointType(joint.get("type", "hinge").upper()),
      transform=SimTransform(position=MJConv.to_pos(self._string2list(joint.get("pos", "0 0 0")))),
    )

  def load_body(self, body : ElementTree.Element) -> SimBody:
    name=body.get("name", "worldbody")
    joints = [SimJoint(type=SimJointType.FIXED, name=name)] # if there are no joints connected, the bodys are welded together
    if body.find("./freejoint") is not None:
      joints = [SimJoint(type=SimJointType.FREE, name=name)]  
    elif len(jObjs := list(body.findall("./joint"))) > 0:
      joints = [self.load_joint(j, name) for j in jObjs]

    joints[0].transform.rotation = self.rotation_from_object(body) # the first joint sets the frame 
    joints[0].transform.position += MJConv.to_pos(self._string2list(body.get("pos", "0 0 0")))
    return SimBody( 
      name=name,
      joints=joints,
      bodies=[self.load_body(b) for b in body.findall("./body")],
      visuals=[visual for geom in body.findall("./geom") + body.findall("./site") if (visual := self.load_visual(geom)) is not None],
    )
  
  def load_worldbody(self):
    worldbody = self._xmlElement.find("./worldbody")
    
    if worldbody is None: return

    self.apply_defaults(worldbody)
    self.worldbody = self.load_body(worldbody)

    
  def apply_defaults(self, element, parent_class : str = "main"):
    
    childclass = element.get("childclass") or parent_class
    cls = element.get("class") or childclass
    
    default = copy.deepcopy(self.default[cls].get(element.tag)) or dict()

    default.update(element.attrib)
    element.attrib = default

    for child in element: self.apply_defaults(child, childclass) # recursive part


  def load_assets(self):
    assets = self._xmlElement.find("./asset")
    if assets is None: return
    self.apply_defaults(assets)
    for asset in assets:
      if asset.tag == "mesh": # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
        file = self.meshdir / asset.attrib["file"]
        scale = self._string2list(asset.get("scale", "1 1 1"))
        mesh, bin_data = MeshLoader.fromFile(file, asset.get("name", file.stem), scale)

        self.meshes.append(mesh)
        self._raw_data[mesh.dataID] = bin_data

      elif asset.tag == "material": # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
        color = self._string2list(asset.get("rgba", "1 1 1 1"))
        material = SimMaterial(
          tag=asset.get("name") or asset.get("type"),
          color=color,
          emissionColor=float(asset.get("emission", "0.0")) * color,   
          specular=float(asset.get("specular", "0.5")),
          shininess=float(asset.get("shininess", "0.5")),
          reflectance=float(asset.get("reflectance", "0.0"))
        )
        
        material.texture = asset.get("texture", None)
        material.texsize = self._string2list(asset.get("texrepeat", "1 1"), dtype=np.int32)
        self.materials.append(material)

      elif asset.tag == "texture": # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
        name = asset.get("name") or asset.get("type")
        builtin = asset.get("builtin") or "none"
        tint = self._string2list(asset.get("rgb1", "1 1 1"))
        if builtin != "none":
          texture, bin_data= TextureLoader.fromBuiltin(name, builtin, tint)
        else:
          file : Path = self.texturedir / asset.get("file")
          texture, bin_data = TextureLoader.fromBytes(name, file.read_bytes(), asset.get("type", "cube"), tint)

        self.textures.append(texture)
        self._raw_data[texture.dataID] = bin_data
      else:
        raise RuntimeError("Invalid asset", asset.tag)

  def __repr__(self):
    return f"<MJCFScene {self._path} meshes={len(self.meshes)}>"