from __future__ import annotations
import abc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Awaitable, Union
import asyncio
from asyncio import sleep as async_sleep
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
import zmq
import zmq.asyncio
import time
import uuid
from json import dumps, loads
import traceback

from .log import logger
from .utils import IPAddress, TopicName, ServiceName, HashIdentifier
from .utils import NodeInfo, INTERNAL_DISCOVER_PORT, EXTERNAL_DISCOVER_PORT, EXTERNAL_BROADCAST_PORT
from .utils import MSG, NodeAddress, NetInfo, NetComponentInfo
from .utils import (
    split_byte,
    get_zmq_socket_port,
    create_address,
    generate_broadcast_msg,
    is_udp_port_in_use,
    is_local_port_in_use,
    search_for_master_node_internal,
)
from .utils import AsyncSocket


class NetInfoManager:

    def __init__(self, local_info: NodeInfo) -> None:
        self.nodes_info: Dict[HashIdentifier, NodeInfo] = {}
        self.net_info: NetInfo = {
            "master": local_info,
            "topics": {name: info for name, info in local_info["topics"].items()},
            "services": {name: info for name, info in local_info["services"].items()},
        }
        self.local_info = local_info
        self.node_id = local_info["nodeID"]

    def get_nodes_info(self) -> Dict[HashIdentifier, NodeInfo]:
        return self.nodes_info

    def get_net_info(self) -> NetInfo:
        return self.net_info

    def check_service(self, name: ServiceName) -> Optional[NetComponentInfo]:
        if name in self.net_info["services"].keys():
            return self.net_info["services"][name]
        return None

    def check_topic(self, name: TopicName) -> Optional[NetComponentInfo]:
        if name in self.net_info["topics"].keys():
            return self.net_info["topics"][name]
        return None

    def register_node(self, info: NodeInfo):
        node_id = info["nodeID"]
        if node_id not in self.nodes_info.keys():
            logger.info(
                f"Node {info['name']} from " f"{info['addr']['ip']} has been launched"
            )
        self.nodes_info[node_id] = info

    def remove_node(self, node_id: HashIdentifier):
        try:
            if node_id in self.nodes_info.keys():
                removed_info = self.nodes_info.pop(node_id)
                logger.info(f"Node {removed_info['name']} is offline")
        except Exception as e:
            logger.error(f"Error occurred when removing node: {e}")

    def get_node_info(self, node_name: str) -> Optional[NodeInfo]:
        for info in self.nodes_info.values():
            if info["name"] == node_name:
                return info
        return None


class NodeManagerBase(abc.ABC):

    manager: Optional[NodeManagerBase] = None

    def __init__(self, node_name: str, host_ip: IPAddress) -> None:
        super().__init__()
        NodeManagerBase.manager = self
        # publisher
        self.pub_socket = self.create_socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:0")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:0")
        self.service_cbs: Dict[bytes, Callable[[bytes], Awaitable]] = {}
        self.local_info: NodeInfo = {
            "name": node_name,
            "nodeID": str(uuid.uuid4()),
            "addr": create_address(host_ip, EXTERNAL_DISCOVER_PORT),
            "type": "Unknown",
            "servicePort": get_zmq_socket_port(self.service_socket),
            "topicPort": get_zmq_socket_port(self.pub_socket),
            "services": [],
            "topics": [],
        }
        self.initialize_local_info()
        self.zmq_context = zmq.asyncio.Context()  # type: ignore
        # start the server in a thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.thread_task)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)
        logger.info(f"Node {self.local_info['name']} is initialized")

    @abc.abstractmethod
    def initialize_local_info(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def check_topic(self, name: TopicName) -> Optional[NetComponentInfo]:
        raise NotImplementedError

    # def create_service(self, service_name: ServiceName) -> AsyncSocket:
    #     return self.create_socket(zmq.REP)

    # def create_topic(self, topic_name: TopicName) -> AsyncSocket:
    #     return self.create_socket(zmq.PUB)

    def create_socket(self, socket_type: int) -> AsyncSocket:
        return self.zmq_context.socket(socket_type)

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.submit_task(self.service_loop)
        self.loop.run_forever()

    def stop_node(self):
        logger.info("Start to stop the node")
        self.running = False
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        self.executor.shutdown(wait=True)

    def thread_task(self):
        logger.info("The node is running...")
        try:
            self.start_event_loop()
        except KeyboardInterrupt:
            self.stop_node()
        except Exception as e:
            logger.error(f"Unexpected error in thread_task: {e}")
        finally:
            logger.info("The node has been stopped")

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def spin(self):
        while True:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt: stopping the node")
                break
        self.stop_node()
        logger.info("The node has been stopped")

    async def service_loop(self):
        logger.info("The service loop is running...")
        service_socket = self.service_socket
        while self.running:
            bytes_msg = await service_socket.recv_multipart()
            service_name, request = split_byte(b"".join(bytes_msg))
            # the zmq service socket is blocked and only run one at a time
            if service_name in self.service_cbs.keys():
                try:
                    await self.service_cbs[service_name](request)
                except asyncio.TimeoutError:
                    logger.error("Timeout: callback function took too long")
                    await service_socket.send(MSG.SERVICE_TIMEOUT.value)
                except Exception as e:
                    logger.error(
                        f"One error occurred when processing the Service "
                        f'"{service_name}": {e}'
                    )
                    traceback.print_exc()
                    await service_socket.send(MSG.SERVICE_ERROR.value)
            await async_sleep(0.01)
        logger.info("Service loop has been stopped")


class NodeManager(NodeManagerBase):
    def initialize_local_info(self):
        self.local_info["type"] = "Node"

    def check_topic(self, name: TopicName) -> Optional[NetComponentInfo]:
        return None

    # TODO: register the node to the master node


class NetManager(NodeManagerBase):

    def __init__(
        self,
        node_name: str,
        host_ip: IPAddress,
        start_broadcast: bool = True,
    ) -> None:
        super().__init__(node_name, host_ip)
        logger.info(
            f"Master Node {node_name} starts at "
            f"{host_ip}:{EXTERNAL_DISCOVER_PORT}"
        )
        if start_broadcast:
            self.start_node_broadcast()

    def initialize_local_info(self):
        self.local_info["type"] = "Master"
        self.net_info_manager = NetInfoManager(self.local_info)

    def start_node_broadcast(self):
        self.submit_task(self.net_service_loop)
        self.submit_task(self.local_node_discover)
        self.submit_task(self.external_node_discover)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    async def local_node_discover(self):
        logger.info("The internal discover loop is running...")
        service_socket = self.create_socket(zmq.PUB)
        service_socket.bind(
            f"tcp://{self.local_info['addr']['ip']}:{INTERNAL_DISCOVER_PORT}"
        )
        while self.running:
            try:
                net_info = self.net_info_manager.get_net_info()
                await service_socket.send(generate_broadcast_msg(net_info))
                await async_sleep(0.5)
            except Exception:
                logger.error("One error occurred when broadcasting")
                traceback.print_exc()
        service_socket.close()
        logger.info("Service loop has been stopped")

    async def external_node_discover(self):
        logger.info("The external discover is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # calculate broadcast ip
        local_info = self.local_info
        _ip = local_info["addr"]["ip"]
        ip_bin = struct.unpack("!I", socket.inet_aton(_ip))[0]
        netmask_bin = struct.unpack("!I", socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack("!I", broadcast_bin))
        while self.running:
            try:
                net_info = self.net_info_manager.get_net_info()
                _socket.sendto(
                    generate_broadcast_msg(net_info),
                    (broadcast_ip, EXTERNAL_DISCOVER_PORT),
                )
            except Exception:
                logger.error("One error occurred when external broadcasting")
                traceback.print_exc()
            await async_sleep(0.5)
        logger.info("Broadcasting has been stopped")

    async def net_service_loop(self):
        logger.info("The net service loop is running...")
        service_socket = self.create_socket(zmq.REP)
        net_service_func = [
            self.register_node_callback,
            self.node_offline_callback,
            self.get_nodes_info_callback,
        ]
        while self.running:
            try:
                bytes_msg = await service_socket.recv_multipart()
                bytes_msg = b"".join(bytes_msg)
                request_name, request = bytes_msg[0], bytes_msg[1:]
                # the zmq service socket is blocked and only run one at a time
                net_service_func[request_name](request.decode())
            except Exception:
                logger.error(
                    "One error occurred when processing the net service"
                )
                traceback.print_exc()
                await service_socket.send(MSG.SERVICE_ERROR.value)
            await service_socket.send(MSG.SERVICE_SUCCESS.value)
        logger.info("Service loop has been stopped")

    def register_node_callback(self, msg: str) -> str:
        # NOTE: something wrong with sending message, but it solved somehow
        client_info: NodeInfo = loads(msg)
        # NOTE: the client info may be updated so the reference cannot be used
        # NOTE: TypeDict is somehow block if the key is not in the dict
        self.net_info_manager.register_node(client_info)
        return "The info has been registered"

    def node_offline_callback(self, msg: str) -> str:
        client_name = msg
        self.net_info_manager.remove_node(client_name)
        return "The info has been removed"

    def get_nodes_info_callback(
        self,
        msg: str,
    ) -> str:
        return dumps(self.net_info_manager.get_nodes_info())

    # def create_service(self, name: ServiceName) -> AsyncSocket:
    #     if self.net_info_manager.check_service(name) is not None:
    #         raise RuntimeError(f"Service {name} has been registered")
    #     service_socket = self.create_socket(zmq.REP)
    #     service_socket.bind(f"tcp://{self.local_info['addr']['ip']}:*")
    #     port = get_zmq_socket_port(service_socket)
    #     service_info: NetComponentInfo = {
    #         "name": name,
    #         "type": "Service",
    #         "addr": create_address(self.local_info["addr"]["ip"], port),
    #         "nodeID": self.local_info["nodeID"],
    #     }
    #     self.local_info["services"].append(name)
    #     self.net_info_manager.net_info["services"][name] = service_info
    #     return service_socket

    # def create_topic(self, topic_name: TopicName) -> AsyncSocket:
    #     if self.net_info_manager.check_topic(topic_name) is not None:
    #         raise RuntimeError("Topic has been registered")
    #     pub_socket = self.create_socket(zmq.PUB)
    #     pub_socket.bind(f"tcp://{self.local_info['addr']['ip']}:*")
    #     port = get_zmq_socket_port(pub_socket)
    #     topic_info: NetComponentInfo = {
    #         "name": topic_name,
    #         "type": "Topic",
    #         "addr": create_address(self.local_info["addr"]["ip"], port),
    #         "nodeID": self.local_info["nodeID"],
    #     }
    #     self.local_info["topics"].append(topic_name)
    #     self.net_info_manager.net_info["topics"][topic_name] = topic_info
    #     return pub_socket

    def check_topic(self, name: IPAddress) -> Optional[NetComponentInfo]:
        return self.net_info_manager.check_topic(name)

    # def remove_local_service(self, service_name: ServiceName) -> None:
    #     if service_name in self.local_info["services"]:
    #         self.local_info["services"].remove(service_name)

    # def remove_local_topic(self, topic_name: TopicName) -> None:
    #     if topic_name in self.local_info["topics"]:
    #         self.local_info["topics"].remove(topic_name)


def init_node(
    node_name: str,
    ip_addr: str = "127.0.0.1",
) -> NodeManagerBase:
    # check if the master node has been initialized in the same program
    if NodeManagerBase.manager is not None:
        return NodeManagerBase.manager

    # check if the master node has been initialized in the same host
    net_info = search_for_master_node_internal(ip_addr)
    if net_info is not None:
        master_name = net_info["master"]["name"]
        logger.info(f"Found master node {master_name} at {ip_addr}")
        return NodeManager(node_name, ip_addr)

    # check if the master node has been initialized in the same subnet
    if is_udp_port_in_use(EXTERNAL_DISCOVER_PORT):
        raise RuntimeError("The external discover port is in use")

    # didn't find any master node, so initialize the master node
    return NetManager(node_name, ip_addr)
