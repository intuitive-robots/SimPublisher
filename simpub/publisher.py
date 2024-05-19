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
  SERVICE_PORT = 5521  # this port is configured in the firewall
  DISCOVERY_PORT = 5520
  STREAMING_PORT = 5522
  def __init__(self, scene : SimScene, discovery_interval : Optional[int] = 2) -> None:
    
    self.discovery_interval = discovery_interval

    self.zmqContext = zmq.Context()  
    self.id = random.randint(100_000, 999_999)

    self.running = False
    self.thread = Thread(target=self._loop)
    self.tracked_joints = dict()

    self.scene = scene
    self.scene_message = serialize_data({
      "id" : scene.id,
      "assets" : [f"{type.value}:{tag}" for type, dict in self.scene.assets.items() for tag in dict],
      "worldbody" : scene.worldbody
    })    


  def start(self):

    discovery_data = dict()
    discovery_data["SERVICE"] = self.SERVICE_PORT
    discovery_data["STREAMING"] = self.STREAMING_PORT
    discovery_message = f"HDAR:{self.id}:{serialize_data(discovery_data)}"

    self.service_thread = ServiceThread(self.zmqContext, port=self.SERVICE_PORT)
    self.streaming_thread = StreamingThread(self.zmqContext, port=self.STREAMING_PORT)
    self.discovery_thread = DiscoveryThread(discovery_message, self.DISCOVERY_PORT, self.discovery_interval)  
    
    self.service_thread.register_action("SCENE_INFO", self._on_scene_request)
    self.service_thread.register_action("ASSET_INFO", self._on_asset_request)
    self.service_thread.register_action("ASSET_DATA", self._on_asset_data_request)

    self.running = True
    self.service_thread.start()
    self.discovery_thread.start()    
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
  
  def update_joint(self, joint_name : str):
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
      msg["data"] = {joint_name : np.array(self.update_joint(joint_name)) for joint_name in self.tracked_joints }
      msg["time"] = time.monotonic()
      self.publish(msg)
