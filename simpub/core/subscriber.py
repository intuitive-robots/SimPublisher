import zmq
from typing import Dict, Callable

from .simpub_manager import SimPubManager, ConnectionAbstract, PortSet

class Subscriber(ConnectionAbstract):
    
    def __init__(self, topic: str):
        super().__init__()
        self.topic: str = topic
        
        self._socket: zmq.Socket = self.manager.zmq_context.socket(zmq.SUB)
        self._callback_func_list: Dict[str, Callable[[str], None]] = {}
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def setting_up_socket(self):
        self._socket.connect(f"tcp://{self.host}:{PortSet.TOOPIC}")

    def execute(self):
        print(f"Start subscribing topic: {self.topic}")
        self.running = True
        while self.running:
            message = self._socket.recv_string()
            topic, msg = message.split(":", 1)
            if topic in self._callback_func_list:
                self._callback_func_list[topic](msg)