
import json
from typing import Callable
import zmq
from simpub.connection.discovery import DiscoveryReceiver
from simpub.connection.streaming import StreamReceiver
from simpub.loaders.json import JsonScene
from simpub.connection.service import RequestService

class SimReceiver:
  DISCOVERY_PORT = 5520
  INSTANCE = None

  def __init__(self):
    self.zmq_context = zmq.Context()
    self.on_init = lambda _: None 
    self.on_update = lambda _: None

  def _none(msg):
    pass

  def start(self):
    self.discovery = DiscoveryReceiver(self._on_discovery, self.DISCOVERY_PORT)
    self.discovery.start()

    self.id = 0
    self.service = RequestService(self.zmq_context)
    self.streaming = StreamReceiver(self.zmq_context, self._on_stream)

  def on(self, event : str):
    def decorator(fn : Callable[[str], None]):
      match event:
        case "INIT":
          self.on_init = fn
        case "UPDATE":
          self.on_update = fn
        case _:
          raise RuntimeError("Invalid function callback")

    return decorator
  
  def request(self, req : str, req_type : type = str):
    return self.service.request(req, req_type=req_type)
  
  def _on_discovery(self, message : str, addr):
    if not message.startswith("HDAR"): return

    _, id, scene = message.split(":", 2)
    if self.id == id: return # same old id so still the same server
  
    if self.service.connected: self.service.disconnect()
    if self.streaming.running: self.streaming.disconnect()
    scene = json.loads(scene)
    self.service_port = scene["SERVICE"]
    self.streaming_port = scene["STREAMING"] 

    self.service.connect(addr, self.service_port)

    new_scene = JsonScene.from_string(self.service.request("SCENE_INFO"))
    self.on_init(new_scene)

    self.streaming.connect(addr, self.streaming_port)
    
  def _on_stream(self, data):
    self.on_update(data)

  def __del__(self):
    self.zmq_context.destroy(0)