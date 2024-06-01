from collections import defaultdict, deque
import copy
from functools import reduce
import operator
import random
from typing import Optional
import numpy as np
from simpub.loaders.asset_loader import MeshLoader, TextureLoader
from simpub.loaders.json import JsonScene
from simpub.simdata import SimBody, SimJoint, SimJointType, SimMaterial, SimScene, SimTransform, SimVisual, SimVisualType
from xml.etree import ElementTree
from pathlib import Path
import math

from simpub.transform import mat2euler, mj2euler, mj2pos, quat2euler

class MJCFFile:

  @staticmethod
  def from_string(content : str, path : Path):
    return MJCFFile(content, path)
  
  @staticmethod
  def from_file(file : Path):
    path : Path = file if isinstance(file, Path) else Path(file)
    return MJCFFile(path.read_text(), path)

  def __init__(self, content : str, path : Path):

    self.root = ElementTree.fromstring(content)
    self.path = path

    self.id = str(random.randint(int(1e9), int(1e10 - 1)))
    
    self.load_includes()

    self.load_compiler()
    
    self.load_defaults()

    self.load_assets()


    self.load_worldbody()

  def load_includes(self):
    while len(includes := [i for i in self.root.iter("include")]):
      for include in includes:
        file = (self.path.parent / include.attrib["file"]).absolute()
        
        if not file.exists(): raise RuntimeError(f"mjcf file {file} not found")

        parsed_include = list(ElementTree.fromstring(file.read_text())) # parse
        parent = {ch: pa for pa in self.root.iter() for ch in pa}[include] # get the parent 
        index = {child : i for i, child in enumerate(parent)}[include] - 1
        parent.remove(include)
        if parent != self.root:
          # Within default or the worlbody we add the children in place
          for i in range(index, index + len(parsed_include)):
            parent.insert(i, parsed_include[i - index])
          continue

        # If were in the root dir we merge the children
        for child in parsed_include:
          if (it := parent.find(child.tag)) is not None and len(list(it)) > 0:
            it.extend(list(child))
          else:
            parent.insert(index, child)

  def load_defaults(self):
    self.default = defaultdict(dict)
    default_container = self.root.find("./default")
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
    self.meshdir = self.path.parent
    self.texturedir = self.path.parent
    compiler = self.root.find("./compiler")

    if compiler is None: return

    self.angle = compiler.get("angle", self.angle)
    self.eulerseq = compiler.get("eulerseq", self.eulerseq)

    self.meshdir    = self.path.parent / compiler.get("meshdir", compiler.get("assetdir", "")) 
    self.texturedir = self.path.parent / compiler.get("texturedir", compiler.get("assetdir", "")) 

  def load_worldbody(self):
    worldbody = self.root.find("./worldbody")
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


    def load_visual(visual : ElementTree.Element) -> Optional[SimVisual]:   
      if int(visual.get("group", 0)) > 2: return None # find a better way to figure this out 
      
      transform = SimTransform(
        rotation=rotation_from_object(visual), 
        position=mj2pos(np.fromstring(visual.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
        scale=np.abs(np.fromstring(visual.get("size", "1 1 1"), dtype=np.float32, sep=' '))
      )

      type = SimVisualType(visual.get("type", "sphere").upper())

      if type == SimVisualType.PLANE:
          transform.scale = np.abs(mj2pos(np.array([transform.scale[0] * 2, transform.scale[1] * 2, 0.001])))
      elif type == SimVisualType.BOX:
          transform.scale = np.abs(mj2pos(transform.scale * 2))
      elif type == SimVisualType.SPHERE:
          transform.scale = np.abs(mj2pos(np.array([transform.scale[0]] * 3)))
      elif type ==  SimVisualType.CYLINDER:
          transform.scale = np.abs(np.array([transform.scale[0], transform.scale[1], transform.scale[0]]))

      return SimVisual(
        type=type,
        transform=transform,
        mesh=visual.get("mesh"),
        material=visual.get("material"),
        color=np.fromstring(visual.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
      )
    
    def load_joint(root : ElementTree.Element):
      ujoint =  SimJoint(
        name=root.get("name", root.tag),
        body=load_body(root),
        transform=SimTransform(
          position=mj2pos(np.fromstring(root.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
          rotation=rotation_from_object(root)
        )
      )


      joints = root.findall("./joint")
      freejoint = root.find("./freejoint")
      if freejoint is not None:
        ujoint.type = SimJointType.FREE
      elif len(joints) > 0: 
        # TODO: There could be multiple joints attached to each body, chaining them together should work 
        # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
        joint = joints[0]
        ujoint.axis = mj2pos(np.fromstring(joint.get("axis", "1 0 0"), dtype=np.int32, sep=' '))
        ujoint.transform.position += mj2pos(np.fromstring(joint.get("pos", "0 0 0"), dtype=np.int32, sep=' '))
        ujoint.name = joint.get("name", ujoint.name)
        ujoint.type = SimJointType(joint.get("type", "hinge").upper())
        ujoint.minrot, ujoint.maxrot = np.degrees(np.fromstring(joint.get("range", "0 0"), dtype=np.float32, sep=' '))

        if ujoint.type is SimJointType.BALL:
          raise RuntimeWarning("Warning: Ball joints in the scene are currently not supported")
      
      return ujoint

    def load_body(body : ElementTree.Element) -> SimBody:
      return SimBody(
        name=body.get("name", "worldbody"),
        joints=[load_joint(b) for b in body.findall("./body")],
        visuals=[visual for geom in body.findall("./geom") if (visual := load_visual(geom)) is not None],
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
    self.meshes = []
    self.materials = []
    self.textures = []
    self.raw_data = dict()
    assets = self.root.find("./asset")
    if assets is None: return
    self.apply_defaults(assets)
    for asset in assets:
      if asset.tag == "mesh":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
          file = self.meshdir / asset.attrib["file"]
          scale = np.array([float(i) for i in asset.get("scale", "1 1 1").split(" ")])
          mesh, bin_data = MeshLoader.fromFile(file, asset.get("name", file.stem), scale)

          self.meshes.append(mesh)
          self.raw_data[mesh.dataID] = bin_data

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
          material.texsize = [int(i) for i in asset.get("texrepeat","1 1").split(" ")]

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
        self.raw_data[texture.dataID] = bin_data
      else:
        raise RuntimeError("Invalid asset", asset.tag)

  def to_scene(self):
    scene = SimScene(
      worldbody=self.worldbody, 
      id=self.id,
      meshes = self.meshes,
      materials = self.materials,
      textures = self.textures,
      _raw_data = self.raw_data
    )


    return scene

  def __repr__(self, indent = 0):
    return f"<MJCFFile {self.path} meshes={len(self.meshes)}>"

if __name__ == "__main__":
  file = "scenes/anybotics_anymal_b/scene.xml"

  print("loading", file)

  mj = MJCFFile(file)
  print(mj)

  scene = mj.to_scene()
  print(JsonScene.to_string(scene))