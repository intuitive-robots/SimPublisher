import zmq
from typing import Callable
from time import sleep

from .net_manager import ConnectionAbstract, PortSet, IPAddress
from .log import logger


class Subscriber(ConnectionAbstract):

    def __init__(self, topic: str, _callback: Callable[[str], None]) -> None:
        super().__init__()
        self.topic: str = topic
        self._callback: Callable[[str], None] = _callback
        self.manager.submit_task(self.wait_for_connection)

    def wait_for_connection(self):
        logger.info(f"Waiting for connection to topic: {self.topic}")
        while self.running:
            sleep(0.01)
            if self.topic not in self.manager.topic_map:
                continue
            target = self.manager.topic_map[self.topic]
            self.manager.topic_callback[self.topic] = self._callback
            if target in self.manager.sub_socket_dict:
                # for host that already connected
                self._socket = self.manager.sub_socket_dict[target]
                break
            # create a new socket for a new host
            self._socket = self.manager.zmq_context.socket(zmq.SUB)
            self.manager.sub_socket_dict[target] = self._socket
            self._socket.connect(f"tcp://{target}:{PortSet.TOOPIC}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.manager.submit_task(self.subscribe_loop, target)
            break
        if self.running:
            logger.info(f"Connected to to topic: {self.topic} at {target}")

    def subscribe_loop(self, host: IPAddress):
        _socket = self.manager.sub_socket_dict[host]
        topic_list = self.manager.host_topic[host]
        topic_callback = self.manager.topic_callback
        while self.running:
            message = _socket.recv_string()
            topic, msg = message.split(":", 1)
            if topic in topic_list:
                topic_callback[topic](msg)

    def on_shutdown(self):
        return super().on_shutdown()
