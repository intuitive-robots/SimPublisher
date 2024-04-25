"""
Implements the streaming connection for the hdar server

How does pub / sub work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
"""

import queue
from typing import Any, Callable, Optional
import zmq
from threading import Thread

class StreamingThread: 
  def __init__(self, context : zmq.Context, port : Optional[int] = None):
    self.thread = Thread(target=self._loop)
    self.zmq_context = context
    self.port = port
    self.running = False
    self.handlers : dict[str, Callable] = dict()
    self.payload = queue.Queue()

  def start(self):
    self.running = True
    self.thread.start()
    
  def stop(self):
    self.running = False
    self.thread.join()
  
  def register(self, tag : str, action : Callable[[zmq.Socket, str, Any], None]):
    self.handlers[tag] = action

  def push(self, tag : str, data):
    if tag not in self.handlers:
      raise RuntimeError("Pushed invalid tag", tag)
    
    self.payload.put((tag, data))

  def _loop(self):
    pub_socket : zmq.Socket = self.zmq_context.socket(zmq.PUB)
    if self.port:
      pub_socket.bind(f"tcp://*:{self.port}")
    else: 
      self.port = pub_socket.bind_to_random_port(f"tcp://*")
    
    print("* Streaming is running on port", self.port)
    while self.running: 
      tag, data = self.payload.get() # This blocks until data is available
        
      self.handlers[tag](pub_socket,tag, data)
