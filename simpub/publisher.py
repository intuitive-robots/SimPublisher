import json
from threading import Thread
from typing import Any, Callable, Optional

from .serialize import serialize_data

from .udata import UAsset, UMesh, UScene
from simpub.connection.discovery import DiscoveryThread
from simpub.connection.streaming import StreamingThread
from simpub.connection.service import ServiceThread

from simpub.model_loader.simscene import SimScene, mj2euler, mj2pos, mj2scale, quat2euler

import zmq
import random 
import time

import numpy as np
import os

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
    

    self.scene = scene.toUScene()

    zmqContext = zmq.Context()  

    
    self.scene_message = serialize_data({
      "id" : scene.id,
      "assets" : list(self.scene.assets.keys()),
      "objects" : scene.objects
    })    

    self.service_thread = ServiceThread(zmqContext, port=service_port)
    
    self.service_thread.register_action("SCENE_INFO", self.on_scene_request)
    self.service_thread.register_action("ASSET_INFO", self.on_asset_request)
    self.service_thread.register_action("ASSET_DATA", self.on_asset_data_request)

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

  def get_scene(self) -> UScene:
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

    value = func(obj) 
    return value

  def on_scene_request(self, socket : zmq.Socket, tag : str):
    socket.send_string(self.scene_message)
  
  def on_asset_request(self, socket : zmq.Socket, tag : str):
    if tag not in self.scene.assets: 
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 
    
    asset : UAsset = self.scene.assets[tag]
    socket.send_string(serialize_data(asset))
  

  def on_asset_data_request(self, socket : zmq.Socket, tag : str):
    if tag not in self.scene.assets:
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 

  
    asset : UMesh = self.scene.assets[tag]
    socket.send(asset._data)

  
  def _loop(self):
    last = 0.0
    while self.running:
      diff = time.monotonic() - last 
      if diff < 1 / self.FPS: 
        time.sleep(1 / self.FPS - diff)

      last = time.monotonic()
      msg = dict()
      msg["data"] = {obj.name : { joint.name : np.array(self.update_joint(joint.name)) for joint in obj.get_joints() } for obj in self.scene.objects}
      msg["time"] = time.monotonic()
      self.publish(msg)
