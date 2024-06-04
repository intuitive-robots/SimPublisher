from collections import defaultdict, deque
import copy
from dataclasses import dataclass
from functools import reduce
import operator
import random
from typing import List, Optional
import numpy as np
from simpub.loaders.asset_loader import MeshLoader, TextureLoader
from simpub.loaders.json import JsonScene
from simpub.simdata import SimBody, SimJoint, SimJointType, SimMaterial, SimScene, SimTransform, SimVisual, SimVisualType
from xml.etree import ElementTree
from pathlib import Path
import math

from simpub.transform import mat2euler, mj2euler, mj2pos, quat2euler

@dataclass
class MJCFScene(SimScene):
  _path : Path = None
  _xmlElement : ElementTree.Element = None
  
  @staticmethod
  def from_string(content : str, path : Path):
    return MJCFScene(
       _xmlElement=ElementTree.fromstring(content), 
       _path=path
    )
  
  @staticmethod
  def from_file(file : Path):
    path : Path = file if isinstance(file, Path) else Path(file)
    return MJCFScene(
      _xmlElement=ElementTree.fromstring(path.read_text()),
      _path=path
    )

  def __post_init__(self):

    if self._path is None or self._xmlElement is None:
      raise RuntimeError("MJCFScene has to instancated from .from_string or .from_file")

    self.id = str(random.randint(int(1e9), int(1e10 - 1)))

    self.merge_includes()

    

    self.load_compiler()
    
    self.load_defaults()

    self.load_assets()

    self.load_worldbody()

  def merge_includes(self):
    while len(includes := list(self._xmlElement.iter("include"))):
      for include in includes:
        file = (self._path.parent / include.attrib["file"]).absolute()
        
        if not file.exists(): raise RuntimeError(f"mjcf file {file} not found")

        parsed_include = list(ElementTree.fromstring(file.read_text())) # parse
        parent = { ch : pa for pa in self._xmlElement.iter() for ch in pa}[include] # get the parent 
        index = {child : i for i, child in enumerate(parent)}[include]
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
    for i in range(1, len(default_container)):
      default_container[0].extend(default_container[i])

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
    self.angle = "degree"
    self.eulerseq = "xyz"
    self.meshdir = self._path.parent
    self.texturedir = self._path.parent
    compiler = self._xmlElement.find("./compiler")

    if compiler is None: return

    self.angle = compiler.get("angle", self.angle)
    self.eulerseq = compiler.get("eulerseq", self.eulerseq)

    self.meshdir    = self._path.parent / compiler.get("meshdir", compiler.get("assetdir", "")) 
    self.texturedir = self._path.parent / compiler.get("texturedir", compiler.get("assetdir", "")) 

  def load_worldbody(self):
    worldbody = self._xmlElement.find("./worldbody")
    if worldbody is None: 
      self.worldbody = None
      return
    self.apply_defaults(worldbody)

    def rotation_from_object(obj : ElementTree.Element) -> np.ndarray: 
      # https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations
      result : np.ndarray
      if quat := obj.get("quat"):
        result = quat2euler(np.fromstring(quat, dtype=np.float32, sep=' '))
      elif axisangle := obj.get("axisangle"):
        raise NotImplementedError()
      elif euler := obj.get("euler"):
        euler = np.fromstring(euler, dtype=np.float32, sep=' ')
        if self.eulerseq == "xyz":
            result = euler
        elif self.eulerseq == "zyx":
            result = euler[::-1]
      elif xyaxes := obj.get("xyaxes"):
        xyaxes = np.fromstring(xyaxes, dtype=np.float32, sep=' ')
        x = xyaxes[:3]
        y = xyaxes[3:6]
        z = np.cross(x, y)
        result = mat2euler(np.array([x, y, z]).T)
      elif zaxis := obj.get("zaxis"):
        raise NotImplementedError()
      else:
        result = np.array([0, 0, 0])

      if self.angle != "degree": result = np.degrees(result)
      return mj2euler(result)


    def load_visual(visual : ElementTree.Element, group : int) -> Optional[SimVisual]:   
      if (vis_group := visual.get("group")) is not None and int(vis_group) != group: return None
      
      transform = SimTransform(
        rotation=rotation_from_object(visual), 
        position=mj2pos(np.fromstring(visual.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
        scale=np.abs(np.fromstring(visual.get("size", "1 1 1"), dtype=np.float32, sep=' '))
      )

      type = SimVisualType(visual.get("type", "sphere").upper())
      if type == SimVisualType.PLANE:
        transform.scale = np.abs(np.array([(transform.scale[0] or 100.0) * 2, 0.001, (transform.scale[1] or 100)* 2]))
      elif type == SimVisualType.BOX:
        transform.scale = np.abs(mj2pos(transform.scale * 2))
      elif type == SimVisualType.SPHERE:
        transform.scale = np.abs(mj2pos(np.array([transform.scale[0] * 2] * 3)))
      elif type ==  SimVisualType.CYLINDER:
        if len(transform.scale) == 3:
          transform.scale = np.abs(np.array([transform.scale[0], transform.scale[1], transform.scale[0]])) 
        else:
          transform.scale = np.abs(np.array([transform.scale[0] * 2, transform.scale[1], transform.scale[0] * 2])) 
      elif type == SimVisualType.CAPSULE:
        if len(transform.scale) == 3:
          transform.scale = np.abs(np.array([transform.scale[0], transform.scale[1], transform.scale[0]])) 
        # elif (fromTo := visual.get("fromto")) is not None:
        #   size = transform.scale[0]
        #   fromTo = np.array([mj2pos(p) for p in np.fromstring(fromTo, np.float32, sep=' ').reshape((2, 3))]) # .view(2, 3)
        #   d = fromTo[1] - fromTo[0]
        #   transform.position = (d / 2) + fromTo[0]

        #   transform.scale = np.array([sum(fromTo[1] - fromTo[0]), size, size])
        #   e1 = np.array([1, 0, 0])
        #   # print(np.linalg.inv(np.outer(e1, d)))
        #   # rotation
        #   R = np.dot(np.dot(np.transpose(e1), np.linalg.inv(np.outer(e1, d))), d)

        #   transform.rotation = np.dot(R, e1)
        #   print("WARNING: fromto property currently not implemented")
        #   # transform.scale = np.abs(np.array([transform.scale[0], size, transform.scale[0]]))

      assert len(transform.scale) == 3 

      return SimVisual(
        type=type,
        transform=transform,
        mesh=visual.get("mesh"),
        material=visual.get("material"),
        color=np.fromstring(visual.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
      )
    
    def load_joint(joint : ElementTree.Element, parent_name : str):
      range = np.fromstring(joint.get("range", "1000000 100000"), dtype=np.float32, sep=' ')
      if self.angle != "degree": range = np.degrees(range)
      return SimJoint(
        name = joint.get("name", parent_name),
        axis=mj2pos(np.fromstring(joint.get("axis", "1 0 0"), dtype=np.int32, sep=' ')),
        minrot = float(range[0]),
        maxrot = float(range[1]),
        transform=SimTransform(position=mj2pos(np.fromstring(joint.get("pos", "0 0 0"), dtype=np.float32, sep=' '))),
        type=SimJointType(joint.get("type", "hinge").upper()),
      )

    def load_body(body : ElementTree.Element) -> SimBody:
      transform=SimTransform(
        position=mj2pos(np.fromstring(body.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
        rotation=rotation_from_object(body)
      )

      group = max(set(int(val) for geom in body.findall("./geom") if (val := geom.get("group") is not None)) or { 0 })

      name=body.get("name", "worldbody")
      joints : List[SimJoint] = list()
      if body.find("./freejoint") is not None:
        joints.append(SimJoint(type=SimJointType.FREE, name=name))  
      elif len(jObjs := list(body.findall("./joint"))) > 0:
        joints.extend(load_joint(j, name) for j in jObjs)
      else:
        joints.append(SimJoint(name))

      joints[0].transform = joints[0].transform + transform
      return SimBody( 
        name=name,
        joints=joints,
        visuals=[visual for geom in body.findall("./geom") + body.findall("./site") if (visual := load_visual(geom, group)) is not None],
        bodies=[load_body(b) for b in body.findall("./body")],
      )
    
    self.worldbody = load_body(worldbody)

    
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
      if asset.tag == "mesh":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
          file = self.meshdir / asset.attrib["file"]
          scale = np.fromstring(asset.get("scale", "1 1 1"), dtype=np.float32, sep=" ")
          mesh, bin_data = MeshLoader.fromFile(file, asset.get("name", file.stem), scale)

          self.meshes.append(mesh)
          self._raw_data[mesh.dataID] = bin_data

      elif asset.tag == "material":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
          color = np.fromstring(asset.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
          material = SimMaterial(
            tag=asset.get("name") or asset.get("type"),
            color=color,
            emissionColor=float(asset.get("emission", "0.0")) * color,   
            specular=float(asset.get("specular", "0.5")),
            shininess=float(asset.get("shininess", "0.5")),
            reflectance=float(asset.get("reflectance", "0.0"))
          )
          
          material.texture = asset.get("texture", None)
          material.texsize = np.fromstring(asset.get("texrepeat", "1 1"), dtype=np.int32, sep=' ')

          self.materials.append(material)
      elif asset.tag == "texture":
        # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
        name = asset.get("name") or asset.get("type")
        builtin = asset.get("builtin", "none")
        tint = np.fromstring(asset.get("rgb1", "1 1 1"), dtype=np.float32, sep=' ')
        if builtin != "none":
          texture, bin_data= TextureLoader.fromBuiltin(name, builtin, tint)
        else:
          file : Path = self.texturedir / asset.get("file")
          texture, bin_data = TextureLoader.fromBytes(name, file.read_bytes(), asset.get("type", "cube"), tint)

        self.textures.append(texture)
        self._raw_data[texture.dataID] = bin_data
      else:
        raise RuntimeError("Invalid asset", asset.tag)

  def __repr__(self, indent = 0):
    return f"<MJCFScene {self._path} meshes={len(self.meshes)}>"

if __name__ == "__main__":
  file = "scenes/anybotics_anymal_b/scene.xml"

  print("loading", file)

  mj = MJCFScene.from_file(file)
  print(mj)

  scene = mj.to_scene()
  print(JsonScene.to_string(scene))