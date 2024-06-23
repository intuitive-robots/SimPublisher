import json
from threading import Thread
from typing import Any, Callable, Optional

from .loaders.json import JsonScene
from .simdata import SimAsset, SimAssetType, SimScene
from simpub.connection.discovery import DiscoverySender
from simpub.connection.streaming import StreamSender
from simpub.connection.service import ReplyService


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
    self.scene_message = JsonScene.to_string(scene)    


  def start(self):

    discovery_data = dict()
    discovery_data["SERVICE"] = self.SERVICE_PORT
    discovery_data["STREAMING"] = self.STREAMING_PORT
    discovery_message = f"HDAR:{self.id}:{json.dumps(discovery_data)}"

    self.service_thread = ReplyService(self.zmqContext, port=self.SERVICE_PORT)
    self.streaming_thread = StreamSender(self.zmqContext, port=self.STREAMING_PORT)
    self.discovery_thread = DiscoverySender(discovery_message, self.DISCOVERY_PORT, self.discovery_interval)  
    
    self.service_thread.register_action("SCENE_INFO", self._on_scene_request)
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

  def register_service(self, tag : str, func : Callable[[zmq.Socket, str], None]):
    self.service_thread.register_action(tag, func)
  
  def update_joint(self, joint_name : str):
    obj, func = self.tracked_joints[joint_name]
    value = func(*obj)
    return value

  def _on_scene_request(self, socket : zmq.Socket, tag : str):
    socket.send_string(self.scene_message)
  
  def _on_asset_data_request(self, socket : zmq.Socket, tag : str):
    if tag not in self.scene._raw_data:
      print("Received invalid data request")
      return
    
    socket.send(self.scene._raw_data[tag])

  def _loop(self):
    last = 0.0
    while len(self.tracked_joints) == 0:
      time.sleep(1)
    while self.running:
      diff = time.monotonic() - last 
      if diff < 1 / self.FPS: 
        time.sleep(1 / self.FPS - diff)

      last = time.monotonic()
      msg = {
        "data" : {joint_name : list(self.update_joint(joint_name)) for joint_name in self.tracked_joints },
        "time" : time.monotonic()
      }
      self.publish(msg)
