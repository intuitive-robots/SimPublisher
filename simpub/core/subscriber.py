import zmq
from typing import Callable
from time import sleep

from .net_manager import ConnectionAbstract, ClientPort, HostInfo
from .log import logger


class Subscriber(ConnectionAbstract):

    def __init__(self, topic: str, callback: Callable[[str], None]) -> None:
        super().__init__()
        self.topic: str = topic
        self._callback: Callable[[str], None] = callback
        self.running = True
        self.manager.submit_task(self.wait_for_connection)

    def check_topic(self) -> HostInfo:
        for client_info in self.manager.clients_info.values():
            if self.topic in client_info["topics"]:
                return client_info
        return None

    def wait_for_connection(self):
        logger.info(f"Waiting for connection to topic: {self.topic}")
        while self.running:
            sleep(0.5)
            target_info = self.check_topic()
            if target_info is None:
                continue
            # for ip which is already connected, only need to set the callback
            self.manager.topic_callback[self.topic] = self._callback
            # create a new socket for a new host
            addr = target_info["ip"]
            if addr in self.manager.sub_socket_dict.keys():
                self._socket = self.manager.sub_socket_dict[addr]
                logger.info(f"Already connected to {addr} for topic: {self.topic}")
            else:
                self._socket = self.manager.zmq_context.socket(zmq.SUB)
                self.manager.sub_socket_dict[addr] = self._socket
                self._socket.connect(f"tcp://{addr}:{ClientPort.TOPIC}")
                self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
                logger.info(f"Connect to {addr} for topic: {self.topic}")
                self.manager.submit_task(self.subscribe_loop, target_info)
            break

    def subscribe_loop(self, host_info: HostInfo):
        _socket = self.manager.sub_socket_dict[host_info["ip"]]
        topic_callback = self.manager.topic_callback
        while self.running:
            message = _socket.recv_string()
            topic, msg = message.split(":", 1)
            topic_callback[topic](msg)

    def on_shutdown(self):
        return super().on_shutdown()
