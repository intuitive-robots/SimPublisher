"""
Implements the service connection for the hdar server

How does req / rep work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
"""

from typing import Any, Callable, Optional
import zmq
from threading import Thread

from udata import UAsset, UAssetType, UMesh, UScene

import json
import numpy as np
import dataclasses as DC


class CustomEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif DC.is_dataclass(obj):
      result = DC.asdict(obj)
      hidden = [key for key in result if key.startswith("_")] 
      for key in hidden:
          del result[key]
      return result
    else:
      return super().default(obj)


class ServiceThread: 
  def __init__(self, context : zmq.Context, scene : UScene, port : Optional[int] = None):
    self.thread = Thread(target=self._loop)
    self._actions : dict = {}
    self.zmq_context = context
    self.port = port
    self.running = True
    self.scene = scene

    self.scene_message = self._data_to_string({
      "id" : scene.id,
      "assets" : [asset for asset in scene.assets],
      "objects" : scene.objects
    }, indent=1)    

    self.register_action("SCENE_INFO", self.on_scene_request)
    self.register_action("ASSET_INFO", self.on_asset_request)
    self.register_action("ASSET_DATA", self.on_asset_data_request)
  

  def start(self):
    if self.scene is None:
      raise RuntimeError("The scene has to be set before the service thread is started")
    self.thread.start()
    
  def stop(self):
    self.running = False
    self.thread.join()
  
  def register_action(self, tag : str, action : Callable[[zmq.Socket, str], None]):
    if tag in self._actions:
      raise RuntimeError("Action was already registred", tag)
    self._actions[tag] = action

  def _loop(self):
    reply_socket : zmq.Socket = self.zmq_context.socket(zmq.REP)
    if self.port:
      reply_socket.bind(f"tcp://*:{self.port}")
    else: 
      self.port = reply_socket.bind_to_random_port(f"tcp://*")
    
    print("* Service is running on port", self.port)
    while self.running: 
      message = reply_socket.recv().decode() # This is blocking so no sleep necessary

      tag, *args = message.split(":", 1)

      
      if tag not in self._actions:
        print(f"Invalid request {tag}")
        reply_socket.send_string("INVALID")
        continue
      
      self._actions[tag](reply_socket, args[0] if args else "")

  def on_scene_request(self, socket : zmq.Socket, tag : str):
    socket.send_string(self.scene_message)
  
  def on_asset_request(self, socket : zmq.Socket, tag : str):
    if tag not in self.scene.assets: 
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 
    
    asset : UAsset = self.scene.assets[tag]
    socket.send_string(self._data_to_string(asset))
  

  def on_asset_data_request(self, socket : zmq.Socket, tag : str):
    if tag not in self.scene.assets:
      print("Received invalid tag", tag)
      socket.send_string("INVALID")
      return 

  
    asset : UMesh = self.scene.assets[tag]
    socket.send(asset._data)

  
  def _data_to_string(self, data : Any, **kwargs):
    return json.dumps(data, cls=CustomEncoder,separators=(",",":"), **kwargs)
  