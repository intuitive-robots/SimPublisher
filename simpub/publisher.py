import json
from threading import Thread
from typing import Any, Callable, Optional

from .serialize import serialize_data

from .simdata import SimAsset, SimAssetType, SimMesh
from simpub.connection.discovery import DiscoveryThread
from simpub.connection.streaming import StreamingThread
from simpub.connection.service import ServiceThread

from simpub.model_loader.simscene import SimScene

import zmq
import random 
import time

import numpy as np

class SimPublisher:
  FPS = 10
  def __init__(
      self, 
      scene : SimScene, 
      discovery_port : int,
      service_port : Optional[int] = None, 
      streaming_port : Optional[int] = None, 
      discovery_interval : Optional[int] = 2
      ) -> None:
    


    zmqContext = zmq.Context()  

    self.scene = scene
    self.scene_message = serialize_data({
      "id" : scene.id,
      "assets" : [f"{type.value}:{tag}" for type, dict in self.scene.assets.items() for tag in dict],
      "worldbody" : scene.worldbody
    })    

    with open("dump.json", "w") as fp:
      fp.write(serialize_data({
      "id" : scene.id,
      "assets" : list(self.scene.assets.keys()),
      "worldbody" : scene.worldbody
    }, indent =2))


    self.service_thread = ServiceThread(zmqContext, port=service_port)
    
    self.service_thread.register_action("SCENE_INFO", self._on_scene_request)
    self.service_thread.register_action("ASSET_INFO", self._on_asset_request)
    self.service_thread.register_action("ASSET_DATA", self._on_asset_data_request)

    self.streaming_thread = StreamingThread(zmqContext, port=streaming_port)

    discovery_data = {
      "SERVICE" : self.service_thread.port,
      "STREAMING" : self.streaming_thread.port,
    }

    self.id = random.randint(100_000, 999_999)

    discovery_message = f"HDAR:{self.id}:{serialize_data(discovery_data)}"
    self.discovery_thread = DiscoveryThread(discovery_message, discovery_port, discovery_interval)  

    self.running = False
    self.thread = Thread(target=self._loop)
    self.tracked_joints = dict()


  def start(self):
    self.service_thread.start()
    self.discovery_thread.start()    
    self.running = True

    self.thread.start()

  def shutdown(self):
    self.discovery_thread.stop()
    self.service_thread.stop()

    self.running = False
    self.thread.join()

  def get_scene(self) -> SimScene:
    return self.scene

  def publish(self, data : Any):
    self.streaming_thread.publish(data)

  def track_joint(self, joint_name : str, obj : Any, func : Any):
    self.tracked_joints[joint_name] = (obj, func)

  def register_service(self, tag : str):

    def decorator(func : Callable[[zmq.Socket, str], None]):
      self.service_thread.register_action(tag, func)
    
    return decorator
  
  def update_joint(self, joint_name : str, return_diff : bool = True):
    if joint_name not in self.tracked_joints: return 0

    obj, func = self.tracked_joints[joint_name]

    value = func(*obj) 
    return value

  def _on_scene_request(self, socket : zmq.Socket, tag : str):
    socket.send_string(self.scene_message)
  
  def _on_asset_request(self, socket : zmq.Socket, tag : str):
    type, name = tag.split(":")
    type = SimAssetType(type)
    if type not in self.scene.assets or name not in self.scene.assets[type]: 
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 
    
    asset : SimAsset = self.scene.assets[type][name]
    socket.send_string(serialize_data(asset))
  
  def _on_asset_data_request(self, socket : zmq.Socket, tag : str):
    type, name = tag.split(":")
    type = SimAssetType(type)
    if type not in self.scene.assets or name not in self.scene.assets[type]:  
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 
    
    asset : SimAsset = self.scene.assets[type][name]
    socket.send(asset._data)

  def _loop(self):
    last = 0.0
    while len(self.tracked_joints) == 0:
      time.sleep(1)
    while self.running:
      diff = time.monotonic() - last 
      if diff < 1 / self.FPS: 
        time.sleep(1 / self.FPS - diff)

      last = time.monotonic()
      msg = dict()
      msg["data"] = {joint.name : np.array(self.update_joint(joint.name)) for joint in self.scene.worldbody.get_joints() }
      msg["time"] = time.monotonic()
      self.publish(msg)
