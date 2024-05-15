"""
Implements the streaming connection for the hdar server

How does pub / sub work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
"""

import queue
from typing import Any, Callable, Optional
from simpub.serialize import serialize_data
import zmq
from threading import Thread

class StreamingThread: 
  def __init__(self, context : zmq.Context, port : Optional[int] = None):
    self.thread = Thread(target=self._loop)
    self.zmq_context = context
    self.port = port
    self.running = False
    self.payload = queue.Queue()

  def start(self):
    self.running = True
    self.thread.start()
    
  def stop(self):
    self.running = False
    self.payload.put("END")
    self.thread.join()
  
  def publish(self, data):
    self.payload.put(data)

  def _loop(self):
    pub_socket : zmq.Socket = self.zmq_context.socket(zmq.PUB)
    if self.port:
      pub_socket.bind(f"tcp://*:{self.port}")
    else: 
      self.port = pub_socket.bind_to_random_port(f"tcp://*")
    
    print("* Streaming is running on port", self.port)
    while self.running: 
      data = self.payload.get() # This blocks until data is available
      pub_socket.send_string(serialize_data(data)) 
    
    
    pub_socket.close()