"""
Implements the service connection for the hdar server

How does req / rep work in zmq:
- https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
"""

from typing import Callable, Optional, Type
from threading import Thread
import zmq


class RequestService:
    def __init__(self, context: zmq.Context) -> None:
        self._actions: dict = {}
        self.zmq_context = context

        self.conn = (None, None)
        self.connected = False

    def connect(self, addr: str, port: int):
        self.req_socket: zmq.Socket = self.zmq_context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{addr}:{port}")
        self.conn = addr, port
        self.connected = True

    def disconnect(self):
        self.req_socket.close(0)
        self.connected = False

    def request(self, req: str, req_type: Type = str):
        self.req_socket.send_string(req)

        if req_type is str:
            return self.req_socket.recv_string()
        if req_type is bytes:
            return self.req_socket.recv()


class ReplyService:
    def __init__(self, context: zmq.Context, port: Optional[int] = None):
        self.thread = Thread(target=self._loop)
        self._actions: dict = {}
        self.zmq_context = context
        self.port = port
        self.running = True

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.reply_socket.close(0)
        self.thread.join()

    def register_action(self, tag: str, action: Callable[[zmq.Socket, str], None]):
        if tag in self._actions:
            raise RuntimeError("Action was already registred", tag)
        self._actions[tag] = action

    def _loop(self):
        self.reply_socket: zmq.Socket = self.zmq_context.socket(zmq.REP)
        if self.port:
            self.reply_socket.bind(f"tcp://*:{self.port}")
        else:
            self.port = self.reply_socket.bind_to_random_port("tcp://*")

        while self.running: 
            message = self.reply_socket.recv().decode()  # This is blocking so no sleep necessary

            tag, *args = message.split(":", 1)
            if tag == "END":
                continue

            if tag not in self._actions:
                print(f"Invalid request {tag}")
                self.reply_socket.send_string("INVALID")
                continue
            self._actions[tag](self.reply_socket, args[0] if args else "")
