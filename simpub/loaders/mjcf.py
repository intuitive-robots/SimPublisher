import io
import math
import random 
from pathlib import Path
import numpy as np
from hashlib import md5

from dm_control.mjcf.base import Element
from dm_control.mjcf.element import RootElement
from dm_control.mjcf.parser import from_path as mjcf_from_path
from dm_control.mjcf.parser import from_xml_string as mjcf_from_xml_string

from dm_control.mjcf.traversal_utils import commit_defaults

from simpub.simdata import *
from simpub.loaders.asset_loader import MeshLoader, TextureLoader

from simpub.transform import mat2euler, mj2euler, mj2pos, mj2scale, quat2euler




class MJCFScene:

  @staticmethod
  def from_string(content : str, assets : dict) -> "SimScene":
    scene = SimScene()
    data : RootElement = mjcf_from_xml_string(content, assets=assets)
    return MJCFScene._load_mjcf(scene, data)

  @staticmethod
  def from_file(path : Path | str) -> "SimScene":

    scene = SimScene()
    
    file_path : Path = path if isinstance(path, Path) else Path(path)
    file_path = file_path.absolute() 

    # scene._meta_data["mjcf_file"] = file_path
    data : RootElement = mjcf_from_path(file_path, resolve_references=True)
    return MJCFScene._load_mjcf(scene, data)

  @staticmethod
  def _load_mjcf(scene : SimScene, data : RootElement):
    
    
    eulerseq = data.compiler.eulerseq or "xyz"
    angle_type = data.compiler.angle or "degrees"

    angle_fn = lambda x: x if angle_type =="degrees" else np.rad2deg(x)



    scene._meta_data["xml_string"] = data.to_xml_string()
    scene._meta_data["xml_assets"] = data.get_assets()

    # commit all the default values
    for tag in { "geom", "mesh", "joint", "body", "material"}: # Add whatever you are using
      for elem in data.find_all(tag):
        if not hasattr(elem, "dclass"): continue  
        commit_defaults(elem)    

    
    for child in data.asset.all_children():
      match child.tag:
        case "mesh":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
          mesh, bin_data = MeshLoader.fromBytes(
            child.name or child.file.prefix, 
            child.file.contents, 
            mesh_type=child.file.extension[1:], 
            scale=child.scale
          )
          scene.meshes.append(mesh)
          scene._raw_data[mesh.dataID] = bin_data
        case "material":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
          color = child.rgba if child.rgba is not None else np.array([1, 1, 1, 1])
          asset = SimMaterial(
            tag=child.name,
            color=color,
            emissionColor=((child.emission or 0.0) * color),   
            specular=child.specular or 0.5,
            shininess=child.shininess or 0.5,
            reflectance=child.reflectance or 0.0
          )

          if child.texture: asset.texture = child.texture.name
          if child.texrepeat is not None: asset.texsize = child.texrepeat

          scene.materials.append(asset)

        case "texture":
          # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
          if child.builtin != "none" and child.builtin is not None:
            texture, bin_data= TextureLoader.fromBuiltin(child.name or child.type, child.builtin, child.rgb1)
          else:
            texture, bin_data = TextureLoader.fromBytes(child.name or child.type, child.file.contents, child.type or "cube", child.rgb1)

          scene.textures.append(texture)
          scene._raw_data[texture.dataID] = bin_data

    def rotation_from_object(obj : Element) -> np.ndarray: # https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations

      quat = getattr(obj, "quat", None)
      axisangle = getattr(obj, "axisangle", None)
      euler = getattr(obj, "euler", None)
      xyaxes = getattr(obj, "xyaxes", None)
      zaxis = getattr(obj, "zaxis", None)

      result : np.ndarray
      if quat is not None:
        result = quat2euler(quat)
      elif axisangle is not None:
        raise NotImplementedError()
      elif euler is not None:
        match eulerseq:
          case "xyz":
            result = euler
          case "zyx":
            result = euler[::-1]
      elif xyaxes is not None:
        x = xyaxes[:3]
        y = xyaxes[3:6]
        z = np.cross(x, y)
        result = mat2euler(np.array([x, y, z]).T)
      elif zaxis is not None:
        raise NotImplementedError()
      else:
        result = np.array([0, 0, 0])

      return mj2euler(angle_fn(result))
      


    def load_visual(visual : Element) -> Optional[SimVisual]:    
      if visual.group is not None and visual.group > 2: return None # find a better way to figure this out 

      type = SimVisualType(visual.type.upper()) if visual.type else SimVisualType.SPHERE

      transform = SimTransform(
        position=mj2pos(visual.pos), 
        rotation=rotation_from_object(visual), 
        scale=np.abs(mj2scale(visual.size))
      )

      return SimVisual(
        type=type,
        transform=transform,
        mesh=(visual.mesh.name or visual.mesh.file.prefix) if type == SimVisualType.MESH else None,
        material=visual.material.name if visual.material else None,
        color=visual.rgba if visual.rgba is not None else np.array([1, 1, 1, 1])
      )
    
    def load_joint(root : Element):
      
      ujoint =  SimJoint(
        name=root.name or root.tag,
        body=load_body(root),
        transform=SimTransform(
          position=mj2pos(root.pos), 
          rotation=rotation_from_object(root)
        )
      )

      joints = root.get_children("joint")
      freejoint = root.get_children("freejoint")

      # TODO: There could be multiple joints attached to each body, chaining them together should work 
      # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
      if freejoint is not None:
        joint = freejoint
        ujoint.type = SimJointType.FREE
      elif len(joints) > 0: 
        joint = joints[0]
        ujoint.axis = mj2pos(joint.axis if joint.axis is not None else np.array([1, 0, 0]))
        ujoint.name = joint.name or ujoint.name
        ujoint.transform.position += mj2pos(joint.pos)
        ujoint.transform.rotation += rotation_from_object(joint)
        ujoint.type = SimJointType((joint.type or "hinge").upper())

        if joint.ref is not None:
          ujoint.initial = joint.ref

        if joint.range is not None:
          ujoint.minrot = math.degrees(joint.range[0])
          ujoint.maxrot = math.degrees(joint.range[1])

        if ujoint.type is SimJointType.BALL:
          raise RuntimeWarning("Warning: Ball joints in the scene are currently not supported")
      

      return ujoint

    def load_body(body : Element) -> SimBody:
      return SimBody(
        name=body.name if hasattr(body, "name") else "worldbody",
        joints=[load_joint(b) for b in body.body],
        visuals=[visual for geom in body.get_children("geom") if (visual := load_visual(geom)) is not None], # this is hacky is there a better way to detect if a geom is collision or visual 
      )
    
    scene.worldbody = load_body(data.worldbody)
    return scene
