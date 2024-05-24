"""
Implements the streaming connection for the hdar server

How does pub / sub work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
"""
from threading import Thread
from typing import Optional, Any
from simpub.loaders.json import JsonScene
import zmq


class StreamReceiver:
  def __init__(self, context : zmq.Context, callback):
    self.zmq_context = context
    self.sub_socket : zmq.Socket = self.zmq_context.socket(zmq.SUB)
    self._thread = Thread(target=self._loop)
    self.running : bool = False
    self.callback = callback
    self.conn = None
    self.connected = False

  def connect(self, addr : str, port : int):
    self.sub_socket.connect(f"tcp://{addr}:{port}")
    self.sub_socket.subscribe("")
    self.running = True
    self.connected = True
    self._thread.start()

  def disconnect(self):
    self.running = False              
    self.sub_socket.close()
    self._thread.join()
    self.connected = False

  def _loop(self):
    while self.running:
      message = self.sub_socket.recv_string()
      self.callback(message)

class StreamSender: 
  def __init__(self, context : zmq.Context, port : Optional[int] = None):
    self.zmq_context = context
    self.port = port
    self.pub_socket : zmq.Socket = self.zmq_context.socket(zmq.PUB)
    if self.port:
      self.pub_socket.bind(f"tcp://*:{self.port}")
    else: 
      self.port = self.pub_socket.bind_to_random_port(f"tcp://*")

    
  def stop(self):
    self.pub_socket.close()
  
  def publish(self, data : Any):
    self.pub_socket.send_string(JsonScene.to_string(data))
