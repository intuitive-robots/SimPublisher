import concurrent
import concurrent.futures
from typing import Dict, Optional
from typing import Callable, Awaitable
import asyncio
from asyncio import sleep as async_sleep
import zmq
import zmq.asyncio
import socket
import time
import uuid
from json import loads

from .log import logger
from .utils import IPAddress, TopicName, ServiceName, HashIdentifier
from .utils import NodeInfo, DISCOVERY_PORT, HEARTBEAT_INTERVAL, MSG
from .utils import calculate_broadcast_addr, split_byte, get_socket_port
from .utils import generate_node_msg, search_for_master_node, create_udp_socket


class NodeInfoManager:

    def __init__(self, local_info: NodeInfo) -> None:
        self.nodes_info: Dict[HashIdentifier, NodeInfo] = {
            local_info["nodeID"].encode(): local_info
        }
        self.local_info = local_info

    def update_nodes_info_dict(
        self, nodes_info: Dict[HashIdentifier, NodeInfo]
    ) -> None:
        self.nodes_info = nodes_info

    def check_node(self, node_id: HashIdentifier) -> bool:
        return node_id in self.nodes_info.keys()

    def get_node(self, node_id: HashIdentifier) -> Optional[NodeInfo]:
        if node_id in self.nodes_info.keys():
            return self.nodes_info[node_id]
        return None

    def check_service(self, service_name: ServiceName) -> bool:
        for node in self.nodes_info.values():
            if service_name in node["serviceList"]:
                return True
        return False

    def check_topic(self, topic_name: TopicName) -> bool:
        for node in self.nodes_info.values():
            if topic_name in node["topicList"]:
                return True
        return False

    def register_service(self, service_name: ServiceName) -> None:
        assert service_name not in self.nodes_info.values()
        for node in self.nodes_info.values():
            node["serviceList"].append(service_name)

    def register_topic(self, topic_name: TopicName) -> None:
        assert topic_name not in self.nodes_info.values()
        for node in self.nodes_info.values():
            node["topicList"].append(topic_name)

    def remove_service(self, service_name: ServiceName) -> None:
        if service_name in self.local_info["serviceList"]:
            self.local_info["serviceList"].remove(service_name)

    def remove_topic(self, topic_name: TopicName) -> None:
        if topic_name in self.local_info["topicList"]:
            self.local_info["topicList"].remove(topic_name)


class MasterNodesInfoManager(NodeInfoManager):

    def __init__(self, local_info: NodeInfo) -> None:
        super().__init__(local_info)
        self.last_heartbeat: Dict[HashIdentifier, float] = {}

    def update_node(self, node_id: HashIdentifier, info: NodeInfo):
        if node_id not in self.nodes_info.keys():
            logger.info(f"Find a new node of {node_id}")
        self.nodes_info[node_id] = info
        self.last_heartbeat[node_id] = time.time()

    def remove_node(self, node_id: HashIdentifier):
        if node_id in self.nodes_info.keys():
            del self.nodes_info[node_id]
            logger.info(f"Node {node_id} has been removed")

    def check_heartbeat(self) -> None:
        for node_id, last in self.last_heartbeat.items():
            if time.time() - last > 2 * HEARTBEAT_INTERVAL:
                self.remove_node(node_id)


class NodeManager:

    manager = None

    def __init__(
        self, host_ip: IPAddress, node_name: str
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
        # heartbeat socket
        self.heartbeat_socket = self.create_socket(zmq.REP)
        self.heartbeat_socket.bind("tcp://*:0")
        # message for broadcasting
        self.local_info: NodeInfo = {
            "name": node_name,
            "nodeID": str(uuid.uuid4()),
            "ip": host_ip,
            "type": "Server",
            "heartbeatPort": get_socket_port(self.heartbeat_socket),
            "servicePort": get_socket_port(self.service_socket),
            "topicPort": get_socket_port(self.pub_socket),
            "serviceList": [],
            "topicList": [],
        }
        # client info

        # start the server in a thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.start_event_loop)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def start_event_loop(self):
        self.nodes_info_manager = NodeInfoManager(self.local_info)
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        # self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)
        self.submit_task(self.heartbeat_loop)
        # self.submit_task(self.check_node_timeout_loop, 1)
        self.loop.run_forever()

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def stop_node(self):
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
                    await service_socket.send_string("Timeout")
                except Exception as e:
                    logger.error(
                        f"One error occurred when processing the Service "
                        f'"{service_name}": {e}'
                    )
                    await service_socket.send(MSG.SERVICE_ERROR.value)
            await async_sleep(0.01)
        logger.info("Service loop has been stopped")

    async def heartbeat_loop(self):
        # set up udp socket
        try:
            _socket = create_udp_socket()
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            broadcast_ip = calculate_broadcast_addr(self.local_info["ip"])
        except Exception as e:
            logger.error(f"Failed to create udp socket: {e}")
            raise Exception("Failed to create udp socket")
        logger.info(
            f"The Net Manager starts broadcasting at {self.local_info['ip']}"
        )
        while self.running:
            try:
                msg = generate_node_msg(self.local_info)
                _socket.sendto(msg, (broadcast_ip, DISCOVERY_PORT))
                await async_sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
        logger.info("Broadcasting loop has been stopped")


class MasterNodeManager(NodeManager):

    def __init__(self, host_ip: IPAddress, node_name: IPAddress) -> None:
        super().__init__(host_ip, node_name)
        self.nodes_info_manager = MasterNodesInfoManager(self.local_info)

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.nodes_info_manager = MasterNodesInfoManager(self.local_info)
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.submit_task(self.service_loop)
        self.submit_task(self.check_heartbeat_loop)
        self.executor.submit(self.node_discover_task)
        self.loop.run_forever()

    def node_discover_task(self):
        logger.info("The Master starts to listen to broadcast")
        try:
            sock = create_udp_socket()
            sock.bind(('0.0.0.0', DISCOVERY_PORT))
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            return
        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                if data == MSG.PING.value:
                    sock.sendto(MSG.PING_ACK.value, addr)
                else:
                    node_id, node_info = split_byte(data)
                    self.nodes_info_manager.update_node(
                        node_id, loads(node_info.decode())
                    )
            except Exception as e:
                logger.error(f"Failed to receive a broadcast: {e}")
            finally:
                time.sleep(0.1)
        self.stop_node()
        logger.info("Node discover task has been stopped")

    async def check_heartbeat_loop(self):
        while self.running:
            for _id, last in self.nodes_info_manager.last_heartbeat.items():
                if time.time() - last > 2 * HEARTBEAT_INTERVAL:
                    self.nodes_info_manager.remove_node(_id)
            await async_sleep(HEARTBEAT_INTERVAL)


def init_node(
    ip_addr: str,
    node_name: str = "PythonNode"
) -> NodeManager:
    # create a udp socket to broadcast for finding a master
    addr = search_for_master_node(ip_addr)
    if NodeManager.manager is not None:
        logger.warning("The node has been initialized")
        return NodeManager.manager
    if addr is None:
        NodeManager.manager = MasterNodeManager(ip_addr, node_name)
    else:
        NodeManager.manager = NodeManager(ip_addr, node_name)
    return NodeManager.manager
