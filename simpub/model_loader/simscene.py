import io
import math
import random 
from pathlib import Path
import numpy as np
from PIL import Image
from hashlib import md5

from dm_control import mjcf

from simpub.udata import *
from simpub.model_loader.mesh_loader import MeshLoader



def mj2quat(quat):
    # note that the order is "[x, y, z, w]"
    return np.array([quat[2], -quat[3], -quat[1], quat[0]], dtype=np.float32)

def mj2pos(pos): 
  if pos is None: return np.array([0, 0, 0])
  return np.array([-pos[1], pos[2], -pos[0]]) 

def mj2euler(rot): 
  if rot is None: return np.array([0, 0, 0])
  return np.array([-rot[1], -rot[2], -rot[0]])

def mj2scale(scale):
  if scale is None: return np.array([1, 1, 1])
  elif len(scale) == 1: return np.array([scale[0], scale[0], scale[0]])
  elif len(scale) == 2: return np.array([scale[0], scale[1], 0])
  return np.array([-scale[1] * 2, scale[2] * 2, scale[0] * 2])

def quat2euler(q):
  if q is None: return np.array([0, 0, 0])
  
  q = q / np.linalg.norm(q)  # Normalize the quaternion
  
  # Calculate the Euler angles
  sin_pitch = 2 * (q[0] * q[2] - q[3] * q[1])
  pitch = np.arcsin(sin_pitch)
  
  if np.abs(sin_pitch) >= 1:
      # Gimbal lock case
      roll = np.arctan2(q[0] * q[1] + q[2] * q[3], 0.5 - q[1]**2 - q[2]**2)
      yaw = 0
  else:
      roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
      yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
  
  return np.array([roll, pitch, yaw])


class SimScene:
  def __init__(self) -> None:
    self.id = str(random.randint(int(1e9), int(1e10 - 1))) # [100.000.000, 999.999.999]
    self.objects : list[UBody] = []
    self.loaded = False
    self.meshes : dict[str, UMesh] = {}
    self.materials : dict[str, UMaterial] = {}
    self.textures : dict[str, UTexture] = {}

    self.xml_string : str
    self.xml_assets : dict

  @staticmethod
  def from_file(path : Path | str) -> "SimScene":

    file_path : Path = path if isinstance(path, Path) else Path(path)
    file_path = file_path.absolute() 

    scene = SimScene()
    
    data : mjcf.RootElement = mjcf.from_path(file_path, resolve_references=True)
    scene._load_mjcf(data)

    return scene

  @staticmethod
  def from_string(content : str, assets : dict) -> "SimScene":
    scene = SimScene()
    
    data : mjcf.RootElement = mjcf.from_xml_string(content, assets=assets)
    scene._load_mjcf(data)

    return scene


  
  def toUScene(self) -> UScene:
    self.loaded = True
    return UScene(
      assets= { 
        **{f"{mesh.type.name}:{mesh.tag}" : mesh for mesh in self.meshes.values() },
        **{f"{material.type.name}:{material.tag}" : material for material in self.materials.values()},
        **{f"{texture.type.name}:{texture.tag}" : texture  for texture in self.textures.values()} 
      },
      id=self.id, 
      objects=self.objects
    )
      
  
  """
  MJCF file loading with dm_control.mjcf
  """
  def _load_mjcf(self, data : mjcf.RootElement):

  
    self.xml_string = data.to_xml_string()
    self.xml_assets = data.get_assets()

    def load_material(material : mjcf.Element):
      color = material.rgba if material.rgba is not None else np.array([1, 1, 1, 1])
      asset = UMaterial(
        tag=material.name,
        color=color,
        emissionColor=((material.emission or 0.0) * color),   
        specular=material.specular or 0.5,
        shininess=material.shininess or 0.5,
        reflectance=material.reflectance or 0.0
      )

      if material.texture: 
        asset.texture = material.texture.name

      self.materials[asset.tag] = asset

    def load_mesh(child : mjcf.Element):

      mesh : UMesh = MeshLoader.fromBytes(child.file.contents, mesh_type=child.file.extension[1:])
      mesh.tag = child.name or child.file.prefix

      for submesh in mesh.submeshes:
        submesh.name = mesh.tag
        if child.scale is not None:
          submesh.transform.scale *= child.scale

        
      self.meshes[mesh.tag] = mesh

    def load_texture(child : mjcf.Element):
      texture = UTexture(tag=child.name or child.type, textype=child.type or "cube")
      
      if child.builtin:
        texture.builtin = child.builtin

      if child.builtin == "none" or child.builtin is None:
        with io.BytesIO(child.file.contents) as data:
          img : Image.Image = Image.open(data)
          img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

          texture.width, texture.height = img.size 
          tex_data = np.frombuffer(img.tobytes(), dtype=np.int8)
          tex_data = tex_data.reshape(texture.width * texture.height, tex_data.shape[0] // (texture.width  * texture.height))

          if tex_data.shape[1] == 3: # upscaling RGB
            rgba = np.zeros((tex_data.shape[0], tex_data.shape[1] + 1), dtype=np.uint8)
            rgba[:, :3] = tex_data  # Copy the RGB values
            rgba[:, 3] = 255  # Set the alpha channel to 
            tex_data = rgba
          texture._data = tex_data.flatten().tobytes()
          texture._data = md5(texture._data).hexdigest()

      self.textures[texture.tag] = texture

    # commit all the default values
    for tag in { "geom", "mesh", "joint", "body", "material"}: # Add whatever you are using
      for elem in data.find_all(tag):
        if not hasattr(elem, "dclass"): continue  
        mjcf.commit_defaults(elem)    

    
    for child in data.asset.all_children():
      match child.tag:
        case "mesh":
          load_mesh(child)
        case "material":
          load_material(child)
        case "texture":
          load_texture(child)

  
    def load_visual(visual : mjcf.Element) -> Optional[UVisual]:    
      if visual.group is not None and visual.group > 2: return None # find a better way to figure this out 

      type = UVisualType(visual.type.upper()) if visual.type else UVisualType.SPHERE
      transform = UTransform(position=mj2pos(visual.pos), rotation=mj2euler(quat2euler(visual.quat)), scale = mj2scale(visual.size))
      return UVisual(
        name=visual.name or visual.tag,
        type=type,
        transform=transform,
        asset=(visual.mesh.name or visual.mesh.file.prefix) if type == UVisualType.MESH else None, # this is a unique file id specified by mujoco
        material=visual.material.name if visual.material else None,
        color=visual.rgba if visual.rgba is not None else np.array([1, 1, 1, 1])
      )
    
    def load_joint(root : mjcf.Element):
      
      ujoint =  UJoint(
        name=root.name or root.tag,
        link=load_link(root),
        transform=UTransform(position=mj2pos(root.pos), rotation=mj2euler(quat2euler(root.quat)))
      )


      joints = root.get_children("joint")
      freejoint = root.get_children("freejoint")
      # TODO: There could be multiple joints attached to each body, chaining them together should work 
      # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
      if freejoint is not None:
        joint = freejoint
        ujoint.type = UJointType.FREE

      elif len(joints) > 0: 
        joint = joints[0]
        ujoint.name = joint.name or ujoint.name
        ujoint.transform.position += mj2pos(joint.pos)

        if hasattr(joint, "quat"):
          ujoint.transform.rotation += mj2euler(quat2euler(joint.quat))

        if joint.range is not None:
          ujoint.minrot = joint.range[0]
          ujoint.maxrot = joint.range[1]
      
        ujoint.type = UJointType((joint.type or "hinge").upper())
        ujoint.axis = mj2pos(joint.axis)

      return ujoint

    def load_link(body : mjcf.Element) -> UBody:
      return UBody(
        name=f"link-{body.name}" if hasattr(body, "name") else "world",
        joints=[load_joint(b) for b in body.body],
        visuals=[visual for geom in body.get_children("geom") if (visual := load_visual(geom)) is not None], # this is hacky is there a better way to detect if a geom is collision or visual 
      )
    
    self.objects += [load_link(data.worldbody)]

