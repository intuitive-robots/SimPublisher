import zmq
from typing import Dict, Callable
import json

from .simpub_manager import ConnectionAbstract, PortSet


class Subscriber(ConnectionAbstract):

    def __init__(self, topic: str):
        super().__init__()
        self.topic: str = topic
        self._callback: Callable[[str], None] = {}

    def setting_up_socket(self):
        return super().setting_up_socket()

    def wait_for_connection(self):
        while self.running:
            if self.topic not in self.manager.topic_map:
                continue
            host = self.manager.topic_map[self.topic]
            if host in self.manager.sub_socket_dict:
                self._socket = self.manager.sub_socket_dict[host]
                break
            self.sub_socket = self.manager.zmq_context.socket(zmq.SUB)
            self.manager.sub_socket_dict[host] = self.sub_socket
            self._socket.connect(f"tcp://{host}:{PortSet.TOOPIC}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def register_callback(
        self,
        _callback: Callable[[str], None]
    ) -> None:
        # TODO: Make it like rospy
        def msg_callback(msg: str):
            json.loads(msg)
            _callback(msg)


    def execute(self):
        print(f"Waiting for topic: {self.topic}")
        self.running = True
        self.wait_for_connection()
        while self.running:
            message = self._socket.recv_string()
            topic, msg = message.split(":", 1)
            if topic in self._callback_func_list:
                self._callback_func_list[topic](msg)
