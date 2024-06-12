
import json
from pathlib import Path
from typing import Any

import numpy as np
import dataclasses as dc

from simpub.simdata import *
# REVIEW: I suggest do the transformation in the beginning of loading data
class _CustomEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.float32):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, Path):
      return str(obj)
    elif dc.is_dataclass(obj):
      return { key : getattr(obj,key) for key in obj.__dataclass_fields__ if not key.startswith("_")}
    else:
      return super().default(obj)
# REVIEW: Don't know what this clase used for from its name, it looks like you want a factory class
# and this class is not necessary, you can just use the SimScene class
# it will make the structure more complex
class JsonScene:

  @staticmethod
  def to_string(data : SimScene):
    return json.dumps(data, separators=(',', ':'), cls=_CustomEncoder)


  @staticmethod 
  def from_string(data : str):
    parsed_scene = json.loads(data)
    meshes=[SimMesh(**mesh) for mesh in parsed_scene.pop("meshes")]
    materials=[SimMaterial(**material) for material in parsed_scene.pop("materials")]
    textures=[SimTexture(**textures) for textures in parsed_scene.pop("textures")]
    worldbody=JsonScene.load_body(parsed_scene.pop("worldbody"))

    return SimScene(
       **parsed_scene,
       meshes=meshes,
       materials=materials,
       textures=textures,
       worldbody=worldbody,
    )

  @staticmethod
  def load_body(data : dict):
    joints=[JsonScene.load_joint(joint) for joint in data.pop("joints")]
    visuals=[JsonScene.load_visual(visual) for visual in data.pop("visuals")]
    return SimBody(
      **data,
      joints=joints,
      visuals=visuals
    )
  
  
  @staticmethod
  def load_visual(data : dict):
    transform=SimTransform(**data.pop("transform"))
    return SimVisual(
       **data,
       transform=transform
    )
  
  @staticmethod
  def load_joint(data : dict):

    transform=SimTransform(**data.pop("transform"))
    body=JsonScene.load_body(data.pop("body"))
    return SimJoint(
       **data,
       transform=transform,
       body=body
    )
