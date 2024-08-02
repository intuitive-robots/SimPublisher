import abc
import enum
from typing import List, Dict, NewType, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import zmq
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
from time import sleep
import json

IPAddress = NewType("IPAddress", str)
Topic = NewType("Topic", str)


class PortSet(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7721
    TOOPIC = 7722


class ConnectionAbstract(abc.ABC):

    def __init__(self):
        self.running: bool = False
        self.manager: SimPubManager = SimPubManager()
        self.host: str = self.manager.host
        self.setting_up_socket()
        SimPubManager().add_connection(self.execute)

    @abc.abstractmethod
    def setting_up_socket(self):
        raise NotImplementedError

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def execute(self):
        raise NotImplementedError

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class SimPubManager:

    _manager = None

    def __new__(cls, *args, **kwargs):
        if not cls._manager:
            cls._manager = super(SimPubManager, cls).__new__(cls, *args, **kwargs)
        return cls._manager

    def __init__(self, host: str = "127.0.0.1"):
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
        self.service_map: Dict[str, IPAddress] = []
        # message for broadcasting
        self.connection_map = {
            "TOPIC": self.topic_map,
            "SERVICE": self.service_map
        }
        self.running: bool = False
        self.executor: ThreadPoolExecutor
        self.futures: List[Future] = []
        self.connections: List[ConnectionAbstract] = []
        self.thread = threading.Thread(target=self.thread_task)
        self.thread.start()

    def join(self):
        self.thread.join()

    def thread_task(self):
        print("Server Tasks has been started")
        with ThreadPoolExecutor(max_workers=5) as executor:
            self.executor = executor
            executor.submit(self.service_loop)
            executor.submit(self.broadcast_loop)
            for connection in self.connections:
                self.futures.append(executor.submit(connection.execute))
        self.running = False

    def add_connection(self, connection: ConnectionAbstract):
        self.connections.append(connection)
        self.executor.submit(connection.execute)

    def subscribe_loop(self, host: IPAddress):
        _socket = self.sub_socket_dict[host]
        topic_list = self.host_topic[host]
        while self.running:
            message = _socket.recv_string()
            topic, msg = message.split(":", 1)
            if topic in topic_list:
                self.topic_callback[topic](msg)

    def service_loop(self):
        while self.running:
            message = self.service_socket.recv_string()
            service, *args = message.split(":", 1)
            if service in self.service_map:
                self.executor.submit(self.service_map[message], *args)
            else:
                self.service_socket.send_string("INVALID")

    def broadcast_loop(self):
        print("The server is broadcasting...")
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

    def register_topic(self, topic: Topic, host: IPAddress):
        if host in self.host_topic:
            pass
        self.topic_map[topic] = host
        self.host_topic[host].append(topic)

    def shutdown(self):
        print("Trying to shutdown server")
        for connection in self.connections:
            connection.shutdown()
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        self.running = False
        self.thread.join()
        print("All the threads have been stopped")
