"""
Implements the service connection for the hdar server

How does req / rep work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
"""

from typing import Any, Callable, Optional
import zmq
from threading import Thread

from simpub.udata import UAsset, UAssetType, UMesh, UScene

import json
import numpy as np
import dataclasses as DC

from simpub.serialize import serialize_data


class ServiceThread: 
  def __init__(self, context : zmq.Context, port : Optional[int] = None):
    self.thread = Thread(target=self._loop)
    self._actions : dict = {}
    self.zmq_context = context
    self.port = port
    self.running = True

  

  def start(self):
    self.thread.start()
    
  def stop(self):
    temp_sock = self.zmq_context.socket(zmq.REQ)
    temp_sock.connect(f"tcp://localhost:{self.port}")
    temp_sock.send_string("END")
    temp_sock.close()

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
      if tag == "END": 
        continue

      if tag not in self._actions:
        print(f"Invalid request {tag}")
        reply_socket.send_string("INVALID")
        continue
      
      self._actions[tag](reply_socket, args[0] if args else "")
    reply_socket.close()

