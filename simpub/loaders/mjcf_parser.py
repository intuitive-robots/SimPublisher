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
  def __init__(self, file : str | Path, tree : set["MJCFFile"] = None):
    self.path : Path = file if isinstance(file, Path) else Path(file)
    self.root = ElementTree.parse(file)
    self.tree = tree or set()
    self.tree.add(self)

    self.id = str(random.randint(int(1e9), int(1e10 - 1)))
    
    self.load_defaults()

    self.load_compiler()

    self.load_assets()

    self.load_includes()

    self.load_worldbody()

  def load_includes(self):
    self.includes = []
    for file in self.root.findall("./include"):
      file = Path(file.attrib["file"])
      file = (self.path.parent / file).absolute()
      
      if not file.exists(): raise RuntimeError(f"mjcf file {file} not found")
      for f in self.tree: 
        if f.path == file: raise RuntimeError(f"mjcf file {file} is contained in a circular include")
      
      self.includes.append(MJCFFile(file, self.tree))

  def load_defaults(self):
    self.default = defaultdict(dict)
    default_container = self.root.find("./default")
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


    self.angle = compiler.attrib.get("angle", self.angle)
    self.eulerseq = compiler.attrib.get("eulerseq", self.eulerseq)

    
    self.meshdir    = self.path.parent / compiler.attrib.get("meshdir", compiler.get("assetdir", "")) 
    self.texturedir = self.path.parent / compiler.attrib.get("texturedir", compiler.get("assetdir", "")) 

  def load_worldbody(self):
    worldbody = self.root.find("./worldbody")

    def rotation_from_object(obj : ElementTree.Element) -> np.ndarray: 
      # https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations
      result : np.ndarray
      if quat := obj.attrib.get("quat"):
        result = quat2euler(np.fromstring(quat, dtype=np.float32, sep=' '))
      elif axisangle := obj.attrib.get("axisangle"):
        print(obj.tag, obj.attrib)
        raise NotImplementedError()
      elif euler := obj.attrib.get("euler"):
        euler = np.fromstring(euler, dtype=np.float32, sep=' ')
        match self.eulerseq:
          case "xyz":
            result = euler
          case "zyx":
            result = euler[::-1]
      elif xyaxes := obj.attrib.get("xyaxes"):
        xyaxes = np.fromstring(xyaxes, dtype=np.float32, sep=' ')
        x = xyaxes[:3]
        y = xyaxes[3:6]
        z = np.cross(x, y)
        result = mat2euler(np.array([x, y, z]).T)
      elif zaxis := obj.attrib.get("zaxis"):
        raise NotImplementedError()
      else:
        result = np.array([0, 0, 0])

      return mj2euler(result if self.angle == "degree" else np.degrees(result))


    def load_visual(visual : ElementTree.Element) -> Optional[SimVisual]:  
      self.apply_defaults(visual)  
      if int(visual.get("group", 0)) > 2: return None # find a better way to figure this out 

      type = SimVisualType(visual.attrib.get("type", "sphere").upper())

      transform = SimTransform(
        position=mj2pos(np.fromstring(visual.attrib.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
        rotation=rotation_from_object(visual), 
        scale=np.abs(np.fromstring(visual.attrib.get("size", "1 1 1"), dtype=np.float32, sep=' '))
      )

      return SimVisual(
        type=type,
        transform=transform,
        mesh=visual.attrib.get("mesh"),
        material=visual.attrib.get("material"),
        color=np.fromstring(visual.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
      )
    
    def load_joint(root : ElementTree.Element):
      self.apply_defaults(root)

      ujoint =  SimJoint(
        name=root.attrib.get("name", root.tag),
        body=load_body(root),
        transform=SimTransform(
          position=mj2pos(np.fromstring(root.attrib.get("pos", "0 0 0"), dtype=np.float32, sep=' ')), 
          rotation=rotation_from_object(root)
        )
      )

      joints = root.findall("./joint")
      freejoint = root.find("./freejoint")
      if freejoint is not None:
        self.apply_defaults(freejoint)
        ujoint.type = SimJointType.FREE
      elif len(joints) > 0: 
        # TODO: There could be multiple joints attached to each body, chaining them together should work 
        # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
        joint = joints[0]
        self.apply_defaults(joint)
        ujoint.axis = mj2pos(np.fromstring(joint.attrib.get("axis", "1 0 0"), dtype=np.int32, sep=' '))
        ujoint.name = joint.get("name", ujoint.name)
        ujoint.type = SimJointType(joint.attrib.get("type", "hinge").upper())

        ujoint.minrot, ujoint.maxrot = np.degrees(np.fromstring(joint.attrib.get("range", "0 0"), dtype=np.float32, sep=' '))

        if ujoint.type is SimJointType.BALL:
          raise RuntimeWarning("Warning: Ball joints in the scene are currently not supported")

      return ujoint

    def load_body(body : ElementTree.Element) -> SimBody:
      self.apply_defaults(body)
      return SimBody(
        name=body.attrib.get("name", "worldbody"),
        joints=[load_joint(b) for b in body.findall("./body")],
        visuals=[visual for geom in body.findall("./geom") if (visual := load_visual(geom)) is not None], # this is hacky is there a better way to detect if a geom is collision or visual 
      )
    self.worldbody = load_body(worldbody)

    
  def apply_defaults(self, element):
    default = self.default[element.get("class", "main")].get(element.tag, dict()).copy()
    for key, value in element.attrib.items():
      default[key] = value
    
    element.attrib = default

  def load_assets(self):
    self.meshes = []
    self.materials = []
    self.textures = []
    self.raw_data = dict()
    includes = self.root.find("./asset")
    if includes is None: return
    for asset in includes:
      self.apply_defaults(asset)
      match asset.tag:
        case "mesh":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
          file = self.meshdir / asset.attrib["file"]
          scale = np.array([float(i) for i in asset.attrib.get("scale", "1 1 1").split(" ")])
          mesh, bin_data = MeshLoader.fromFile(file, asset.attrib.get("name", file.stem), scale)

          self.meshes.append(mesh)
          self.raw_data[mesh.dataID] = bin_data

        case "material":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
          color = np.fromstring(asset.attrib.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
          material = SimMaterial(
            tag=asset.attrib.get("name") or asset.attrib.get("type"),
            color=color,
            emissionColor=float(asset.attrib.get("emission", "0.0")) * color,   
            specular=float(asset.attrib.get("specular", "0.5")),
            shininess=float(asset.attrib.get("shininess", "0.5")),
            reflectance=float(asset.attrib.get("reflectance", "0.0"))
          )
          
          material.texture = asset.attrib.get("texture", None)
          material.texsize = [int(i) for i in asset.attrib.get("texrepeat","1 1").split(" ")]

          self.materials.append(material)
        case "texture":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
          name = asset.attrib.get("name") or asset.attrib.get("type")
          builtin = asset.attrib.get("builtin", "none")
          tint = np.fromstring(asset.attrib.get("rgb1", "1 1 1"), dtype=np.float32, sep=' ')
          if builtin != "none":
            texture, bin_data= TextureLoader.fromBuiltin(name, builtin, tint)
          else:
            file : Path = self.texturedir / asset.attrib.get("file")
            texture, bin_data = TextureLoader.fromBytes(name, file.read_bytes(), asset.attrib.get("type", "cube"), tint)

          self.textures.append(texture)
          self.raw_data[texture.dataID] = bin_data
        case _:
          raise RuntimeError("Invalid asset", asset.tag)

  def to_scene(self):
    scene = SimScene(
      worldbody=SimBody("worldbody", list(), list()), 
      id=self.id
    )

    for mjcf in self.tree:
      scene.meshes += mjcf.meshes
      scene.materials += mjcf.materials
      scene.textures += mjcf.textures
      scene._raw_data.update(mjcf.raw_data)
      
      scene.worldbody.visuals += mjcf.worldbody.visuals
      scene.worldbody.joints += mjcf.worldbody.joints

    return scene

  def __repr__(self, indent = 0):
    string = " " * (5  * indent) + f"<MJCFFile {self.path} meshes={len(self.meshes)}>"
    for i in self.includes:
      string += "\n" + i.__repr__(indent + 1)
    return string
    
if __name__ == "__main__":
  file = "scenes/anybotics_anymal_b/scene.xml"

  print("loading", file)

  mj = MJCFFile(file)
  print(mj)

  scene = mj.to_scene()
  print(JsonScene.to_string(scene))