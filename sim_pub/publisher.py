import json
from typing import Optional

from sim_pub.udata import UScene
from .connection.discovery import DiscoveryThread
from .connection.streaming import StreamingThread
from .connection.service import ServiceThread

from .model_loader.asset import AssetLoader
from .simulator import PublisherSimulator

from pathlib import Path
import zmq
import random 

class SimPublisher:

  def __init__(
      self, 
      scene_path : Path | str, 
      discovery_port : int,
      service_port : Optional[int] = None, 
      streaming_port : Optional[int] = None, 
      discovery_interval : Optional[int] = 2
      ) -> None:
    if isinstance(scene_path, str): scene_path = Path(scene_path)
    
    self.asset_loader = AssetLoader()

    self.asset_loader.load_asset(scene_path)

    self.scene = self.asset_loader.package()

    zmqContext = zmq.Context()

    self.service_thread = ServiceThread(zmqContext, self.scene, port=service_port)

    self.streaming_thread = StreamingThread(zmqContext, port=streaming_port)

    discovery_data = {
      "SERVICE" : self.service_thread.port,
      "STREAMING" : self.streaming_thread.port,
    }

    self.id = random.randint(100_000, 999_999)

    discovery_message = f"HDAR:{self.id}:{json.dumps(discovery_data, separators=(',', ':'))}"

    self.discovery_thread = DiscoveryThread(discovery_message, discovery_port, discovery_interval)  
  
  def start(self):
    self.service_thread.start()
    self.streaming_thread.start()
    self.discovery_thread.start()

  def shutdown(self):
    self.discovery_thread.stop()
    self.streaming_thread.stop()
    self.service_thread.stop()

  def get_scene(self) -> UScene:
    return self.scene