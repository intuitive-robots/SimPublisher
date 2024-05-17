"""
Implements the streaming connection for the hdar server

How does pub / sub work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html
"""
from typing import Optional, Any
from simpub.serialize import serialize_data
import zmq

class StreamingThread: 
  def __init__(self, context : zmq.Context, port : Optional[int] = None):
    self.zmq_context = context
    self.port = port
    self.pub_socket : zmq.Socket = self.zmq_context.socket(zmq.PUB)
    if self.port:
      self.pub_socket.bind(f"tcp://*:{self.port}")
    else: 
      self.port = self.pub_socket.bind_to_random_port(f"tcp://*")

    
  def __del__(self):
    self.pub_socket.close()
  
  def publish(self, data : Any):
    self.pub_socket.send_string(serialize_data(data))
