"""
Implements the service connection for the hdar server

How does req / rep work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
"""

from typing import Callable, Optional, Type
from threading import Thread
import zmq

class RequestService:
  def __init__(self, context : zmq.Context) -> None:
    self._actions : dict = {}
    self.zmq_context = context
    self.running = True  

    self.conn = (None, None)
    self.req_socket : zmq.Socket = self.zmq_context.socket(zmq.REQ)
    self.connected = False
  
  def connect(self, addr : str, port : int):
    self.conn = addr, port
    self.req_socket.connect(f"tcp://{self.conn[0]}:{self.conn[1]}")
    self.connected = True
  
  def disconnect(self):
    self.req_socket.close()
    self.connected = False
  
  def request(self, req : str, req_type : Type = str):
    self.req_socket.send_string(req)

    if req_type is str:
      return self.req_socket.recv_string()
    if req_type is bytes:
      return self.req_socket.recv()
    


class ReplyService: 
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

