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
import uuid
from .log import logger

IPAddress = NewType("IPAddress", str)
Topic = NewType("Topic", str)
Service = NewType("Service", str)


class ServerPort(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7721
    TOPIC = 7722


class ClientPort(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7723
    TOPIC = 7724


class HostInfo(TypedDict):
    name: str
    ip: IPAddress
    topics: List[Topic]
    services: List[Service]


class ConnectionAbstract(abc.ABC):

    def __init__(self):
        self.running: bool = False
        self.manager: NetManager = NetManager.manager
        self.host_ip: str = self.manager.local_info["ip"]
        self.host_name: str = self.manager.local_info["host"]

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class NetManager:

    manager = None

    def __init__(
        self,
        host_ip: IPAddress = "127.0.0.1",
        host_name: str = "SimPub"
    ) -> None:
        NetManager.manager = self
        self._initialized = True
        self.zmq_context = zmq.Context()
        # subscriber
        self.sub_socket_dict: Dict[IPAddress, zmq.Socket] = {}
        self.topic_callback: Dict[Topic, Callable] = {}
        # publisher
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:{ServerPort.TOPIC}")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:{ServerPort.SERVICE}")
        self.service_callback: Dict[str, Callable] = {}
        # message for broadcasting
        self.local_info = HostInfo()
        self.local_info["host"] = host_name
        self.local_info["ip"] = host_ip
        self.local_info["topics"] = []
        self.local_info["services"] = []
        # host info
        self.clients_info: Dict[str, HostInfo] = {}
        # setting up thread pool
        self.running: bool = True
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=5)
        self.futures: List[Future] = []
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)

    def submit_task(self, task: Callable, *args):
        future = self.executor.submit(task, *args)
        self.futures.append(future)

    def join(self):
        for future in self.futures:
            future.result()
        self.executor.shutdown()

    def service_loop(self):
        logger.info("The service is running...")
        self.manager.register_local_service(
            "Register", self.register_client_callback
        )
        while self.running:
            message = self.service_socket.recv_string()
            service, request = message.split(":", 1)
            if service in self.service_callback.keys():
                # the zmq service socket is blocked and only run one at a time
                self.service_callback[service](request, self.service_socket)
            else:
                self.service_socket.send_string("Invild Service")

    def broadcast_loop(self):
        logger.info("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        _id = str(uuid.uuid4())
        # calculate broadcast ip
        local_info = self.local_info
        ip_bin = struct.unpack('!I', socket.inet_aton(local_info["ip"]))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))
        while self.running:
            msg = f"SimPub:{_id}:{json.dumps(local_info)}"
            _socket.sendto(msg.encode(), (broadcast_ip, ServerPort.DISCOVERY))
            sleep(0.5)
        logger.info("Broadcasting has been stopped")

    def register_client_callback(self, msg: str, socket: zmq.Socket):
        # NOTE: something woring with sending message, but it solved somehow
        socket.send_string("The info has been registered")
        client_info: HostInfo = json.loads(msg)
        client_name = client_info["name"]
        # NOTE: the client info may be updated so the reference cannot be used
        # NOTE: TypeDict is somehow block if the key is not in the dict
        self.clients_info[client_name] = client_info
        logger.info(f"Host {client_name} has been registered")

    def register_local_topic(self, topic: Topic):
        if topic in self.local_info["topics"]:
            logger.warning(f"Host {topic} is already registered")
        self.local_info["topics"].append(topic)

    def register_local_service(
        self, service: str, callback: Callable
    ) -> None:
        self.local_info["services"].append(service)
        self.service_callback[service] = callback

    def shutdown(self):
        logger.info("Shutting down the server")
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        self.running = False
        logger.info("Server has been shut down")


def init_net_manager(host: str):
    if NetManager.manager is not None:
        return NetManager.manager
    return NetManager(host)
