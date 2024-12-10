import concurrent
import concurrent.futures
from typing import List, Dict, Optional
from typing import Callable, TypedDict, Awaitable
import asyncio
from asyncio import sleep as async_sleep
import zmq
import zmq.asyncio
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import time
import uuid
from json import dumps, loads

from .log import logger
from .utils import calculate_broadcast_addr, split_byte, get_socket_port

IPAddress = str
TopicName = str
ServiceName = str
AsyncSocket = zmq.asyncio.Socket
HashIdentifier = bytes

BROADCAST_INTERVAL = 0.1
DISCOVERY_PORT = 7720


class NodeInfo(TypedDict):
    name: str
    identifier: str  # hash code since bytes is not JSON serializable
    ip: str
    type: str
    servicePort: str
    topicPort: str
    serviceList: List[ServiceName]
    topicList: List[TopicName]


class NodeInfoManager:

    def __init__(self, local_info: NodeInfo) -> None:
        self.node_info: Dict[HashIdentifier, NodeInfo] = {
            local_info["identifier"].encode(): local_info
        }
        self.local_info = local_info

    def update_node(self, identifier: HashIdentifier, info: NodeInfo):
        self.node_info[identifier] = info

    def remove_node(self, identifier: HashIdentifier):
        if identifier in self.node_info.keys():
            del self.node_info[identifier]

    def check_node(self, identifier: HashIdentifier) -> bool:
        return identifier in self.node_info.keys()

    def get_node(self, identifier: HashIdentifier) -> Optional[NodeInfo]:
        if identifier in self.node_info.keys():
            return self.node_info[identifier]
        return None

    def check_service(self, service_name: ServiceName) -> bool:
        for node in self.node_info.values():
            if service_name in node["serviceList"]:
                return True
        return False

    def check_topic(self, topic_name: TopicName) -> bool:
        for node in self.node_info.values():
            if topic_name in node["topicList"]:
                return True
        return False

    def register_service(self, service_name: ServiceName) -> None:
        assert service_name not in self.node_info.values()
        for node in self.node_info.values():
            node["serviceList"].append(service_name)

    def register_topic(self, topic_name: TopicName) -> None:
        assert topic_name not in self.node_info.values()
        for node in self.node_info.values():
            node["topicList"].append(topic_name)

    def remove_service(self, service_name: ServiceName) -> None:
        if service_name in self.local_info["serviceList"]:
            self.local_info["serviceList"].remove(service_name)

    def remove_topic(self, topic_name: TopicName) -> None:
        if topic_name in self.local_info["topicList"]:
            self.local_info["topicList"].remove(topic_name)


class NodeManager:

    manager = None

    def __init__(
        self, host_ip: IPAddress = "127.0.0.1", node_name: str = "PythonNode"
    ) -> None:
        NodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context()  # type: ignore
        # # subscriber
        # self.sub_socket_dict: Dict[IPAddress, AsyncSocket] = {}
        # publisher
        self.pub_socket = self.create_socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:0")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind("tcp://*:0")
        self.service_cbs: Dict[bytes, Callable[[bytes], Awaitable[bytes]]] = {}
        # message for broadcasting
        self.local_info: NodeInfo = {
            "name": node_name,
            "identifier": str(uuid.uuid4()),
            "ip": host_ip,
            "type": "Server",
            "servicePort": get_socket_port(self.service_socket),
            "topicPort": get_socket_port(self.pub_socket),
            "serviceList": [],
            "topicList": [],
        }
        # client info
        self.node_info_manager = NodeInfoManager(self.local_info)
        # start the server in a thread pool
        self.running = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.server_future = self.executor.submit(self.start_event_loop)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.05)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def start(self):
        self.running = True

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.register_loop)
        self.submit_task(self.service_loop)
        while not self.running:
            time.sleep(0.01)
        self.loop.run_forever()

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def stop_server(self):
        if not self.running:
            return
        if not self.loop:
            return
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        self.spin()

    def spin(self):
        self.executor.shutdown(wait=True)

    async def service_loop(self):
        logger.info("The service is running...")
        while self.running:
            bytes_msg = await self.service_socket.recv()
            service_name, request = split_byte(bytes_msg)
            # the zmq service socket is blocked and only run one at a time
            if service_name in self.service_cbs.keys():
                try:
                    await self.service_cbs[service_name](request)
                except asyncio.TimeoutError:
                    logger.error("Timeout: callback function took too long")
                    await self.service_socket.send_string("Timeout")
                except Exception as e:
                    logger.error(
                        f"One error occurred when processing the Service "
                        f'"{service_name}": {e}'
                    )
                    # TODO: standard error message
                    await self.service_socket.send(b"Error")
            await async_sleep(0.01)

    async def broadcast_loop(self):
        logger.info(
            f"The Net Manager starts broadcasting at {self.local_info['ip']}"
        )
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        broadcast_ip = calculate_broadcast_addr(self.local_info["ip"])
        address = (broadcast_ip, DISCOVERY_PORT)
        identifier = self.local_info["identifier"].encode()
        while self.running:
            try:
                local_info = self.local_info  # update local info
                msg = b"".join([identifier, b":", dumps(local_info).encode()])
                _socket.sendto(msg, address)
                await async_sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
        logger.info("Broadcasting has been stopped")

    async def register_loop(self):
        try:
            loop = asyncio.get_event_loop()
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
                socket.IPPROTO_UDP
            )
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(False)
            sock.bind(('0.0.0.0', DISCOVERY_PORT))
        except Exception as e:
            logger.info(f"[ERROR] Failed to create socket: {e}")
            return
        while self.running:
            try:
                msgpack = await loop.sock_recv(sock, 1024)
                identifier, msg = split_byte(msgpack)
                if not self.node_info_manager.check_node(identifier):
                    logger.info(f"Found node: {identifier}")
                node_info: NodeInfo = loads(msg.decode())
                self.node_info_manager.update_node(identifier, node_info)
            except Exception as e:
                print(f"[ERROR] Failed to receive broadcast: {e}")


def init_net_manager(host: str) -> NodeManager:
    if NodeManager.manager is not None:
        return NodeManager.manager
    return NodeManager(host)
