import abc
import enum
from typing import List, Dict, NewType, Callable, TypedDict
from concurrent.futures import ThreadPoolExecutor, Future
import zmq
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
from time import sleep
import json

from .log import logger

IPAddress = NewType("IPAddress", str)
Topic = NewType("Topic", str)


class PortSet(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7721
    TOOPIC = 7722


class ClientInfo(TypedDict):
    Host: IPAddress
    Topics: List[Topic]


class ConnectionAbstract(abc.ABC):

    def __init__(self):
        self.running: bool = False
        self.manager: SimPubManager = SimPubManager()
        self.host: str = self.manager.host

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class SimPubManager:

    _manager = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._manager is None:
            cls._manager = super().__new__(cls, *args, **kwargs)
        return cls._manager

    def __init__(self, host: str = "127.0.0.1"):
        if self._initialized:
            return
        self._initialized = True
        self.host: IPAddress = host
        self.zmq_context = zmq.Context()
        # subscriber
        self.sub_socket_dict: Dict[IPAddress, zmq.Socket] = {}
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.topic_map: Dict[Topic, IPAddress] = {}
        self.host_topic: Dict[IPAddress, List[Topic]] = {}
        self.topic_callback: Dict[Topic, Callable] = {}
        # publisher
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host}:{PortSet.TOOPIC}")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host}:{PortSet.SERVICE}")
        self.service_callback: Dict[str, Callable] = {}
        self.service_map: Dict[str, IPAddress] = []
        # message for broadcasting
        self.connection_map = {
            "TOPIC": self.topic_map,
            "SERVICE": self.service_map
        }
        # setting up thread pool
        self.running: bool = True
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=5)
        self.futures: List[Future] = []
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)

    def submit_task(self, task: Callable):
        future = self.executor.submit(task)
        self.futures.append(future)

    def join(self):
        for future in self.futures:
            future.result()
        self.executor.shutdown()

    def service_loop(self):
        logger.info("The service is running...")
        self.service_map["Register"] = self.register_client
        while self.running:
            message = self.service_socket.recv_string()
            service, msg = message.split(":", 1)
            if service in self.service_map:
                self.executor.submit(self.service_map[service], msg)
            else:
                self.service_socket.send_string("INVALID")

    def broadcast_loop(self):
        logger.info("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # calculate broadcast ip
        ip_bin = struct.unpack('!I', socket.inet_aton(self.host))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))
        while self.running:
            msg = f"SimPub:{json.dumps(self.connection_map)}"
            _socket.sendto(msg.encode(), (broadcast_ip, PortSet.DISCOVERY))
            sleep(0.5)
        logger.info("Broadcasting has been stopped")

    def register_client(self, msg: str):
        info: ClientInfo = json.loads(msg)
        for topic in info["Topics"]:
            self.register_topic(topic, info["Host"])

    def register_topic(self, topic: Topic, host: IPAddress):
        if host in self.host_topic:
            pass
        self.topic_map[topic] = host
        self.host_topic[host].append(topic)

    def shutdown(self):
        logger.info("Shutting down the server")
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        self.running = False
        logger.info("Server has been shut down")
