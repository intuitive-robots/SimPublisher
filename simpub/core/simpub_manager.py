import abc
import enum
from typing import List, Dict, NewType
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import zmq
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
from time import sleep
import json

IPAddress = NewType("IPAddress", str)


class PortSet(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7721
    TOOPIC = 7722


class ConnectionAbstract(abc.ABC):
    
    def __init__(self):
        self.running: bool = False
        self.manager: SimPubManager = SimPubManager()
        self.host: str = self.manager.host
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

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SimPubManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, host: str = "127.0.0.1"):
        self.host: str = host
        self.zmq_context = zmq.Context()
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host}:{PortSet.TOOPIC}")
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host}:{PortSet.SERVICE}")
        self.topic_map: Dict[str, IPAddress] = {}
        self.service_map: Dict[str, IPAddress] = []
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


    def broadcast(self):
        print("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # calculate broadcast ip
        ip_bin = struct.unpack('!I', socket.inet_aton(self.host))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        self.broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))
        while self.running:
            msg = f"SimPub:{json.dumps(self.connection_map)}"
            _socket.sendto(msg.encode(), (self.broadcast_ip, PortSet.DISCOVERY))
            sleep(0.5)

    def generate_upd_msg(self) -> bytes:
        return 

    def join(self):
        self.thread.join()

    def thread_task(self):
        print("Server Tasks has been started")
        with ThreadPoolExecutor(max_workers=5) as executor:
            self.executor = executor
            executor.submit(self.broadcast)
            for connection in self.connections:
                self.futures.append(executor.submit(connection.execute))
        self.running = False

    def shutdown(self):
        print("Trying to shutdown server")
        for connection in self.connections:
            connection.shutdown()
        self.thread.join()
        print("All the threads have been stopped")

    def add_connection(self, connection: ConnectionAbstract):
        self.connections.append(connection)
        self.executor.submit(connection.execute)
        
