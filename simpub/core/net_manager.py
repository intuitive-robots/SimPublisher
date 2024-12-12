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
from json import loads, dumps
import traceback

from .log import logger
from .utils import IPAddress, Address, TopicName, ServiceName, HashIdentifier
from .utils import NodeInfo, DISCOVERY_PORT, HEARTBEAT_INTERVAL
from .utils import EchoHeader, MSG
from .utils import calculate_broadcast_addr, split_byte, get_zmq_socket_port
from .utils import generate_node_msg, search_for_master_node, create_udp_socket
from .utils import split_byte_to_str


class NodeInfoManager:

    def __init__(self, local_info: NodeInfo) -> None:
        self.nodes_info: Dict[HashIdentifier, NodeInfo] = {
            local_info["nodeID"]: local_info
        }
        self.local_info = local_info

    def update_nodes_info(
        self, nodes_info_dict: Dict[HashIdentifier, NodeInfo]
    ) -> None:
        self.nodes_info = nodes_info_dict

    def get_nodes_info_msg(self) -> bytes:
        return dumps(self.nodes_info).encode()

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

    def remove_local_service(self, service_name: ServiceName) -> None:
        if service_name in self.local_info["serviceList"]:
            self.local_info["serviceList"].remove(service_name)

    def remove_local_topic(self, topic_name: TopicName) -> None:
        if topic_name in self.local_info["topicList"]:
            self.local_info["topicList"].remove(topic_name)


class MasterNodesInfoManager(NodeInfoManager):

    def __init__(self, local_info: NodeInfo) -> None:
        super().__init__(local_info)
        self.last_heartbeat: Dict[HashIdentifier, float] = {}

    def update_node(self, node_id: HashIdentifier, info: NodeInfo):
        if node_id not in self.nodes_info.keys():
            logger.info(
                f"Node {info['name']} has been launched"
            )
        self.nodes_info[node_id] = info
        self.last_heartbeat[node_id] = time.time()

    def remove_node(self, node_id: HashIdentifier):
        try:
            if node_id in self.nodes_info.keys():
                removed_info = self.nodes_info.pop(node_id)
                logger.info(f"Node {removed_info['name']} has been removed")
                self.last_heartbeat.pop(node_id)
        except Exception as e:
            logger.error(f"Error occurred when removing node: {e}")

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
            # "master": False,
            "type": "Server",
            # "heartbeatPort": get_socket_port(self.heartbeat_socket),
            "servicePort": get_zmq_socket_port(self.service_socket),
            "topicPort": get_zmq_socket_port(self.pub_socket),
            "serviceList": [],
            "topicList": [],
        }
        logger.info(f"Node {self.local_info['name']} is initialized")
        # start the server in a thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.start_event_loop)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def thread_task(self):
        try:
            self.start_event_loop()
        except KeyboardInterrupt:
            self.stop_node()
        finally:
            logger.info("The node has been stopped")

    def start_event_loop(self):
        self.nodes_info_manager = NodeInfoManager(self.local_info)
        self.loop = asyncio.new_event_loop()
        self.running = True
        self.connected = False
        asyncio.set_event_loop(self.loop)
        self.submit_task(self.service_loop)
        self.submit_task(self.node_task)
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
        self.running = False
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        self.spin(False)

    def spin(self, wait=True):
        self.executor.shutdown(wait=wait)

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
                    await service_socket.send(MSG.SERVICE_ERROR.value)
            await async_sleep(0.01)
        logger.info("Service loop has been stopped")

    async def node_task(self):
        try:
            _socket = create_udp_socket()
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            broadcast_ip = calculate_broadcast_addr(self.local_info["ip"])
            _socket.bind((self.local_info["ip"], 0))
        except Exception as e:
            logger.error(f"Failed to create udp socket: {e}")
            raise Exception("Failed to create udp socket")
        while self.running:
            master_address = await self.search_for_master_node(
                _socket, broadcast_ip
            )
            if master_address is not None:
                await self.heartbeat_loop(_socket, master_address)
            await async_sleep(0.1)

    async def search_for_master_node(
        self,
        _socket: socket.socket,
        broadcast_ip: IPAddress,
        time_out: float = 0.1
    ) -> Optional[Address]:
        loop = asyncio.get_running_loop()
        master_address = None
        logger.info("Searching for the master node")
        while self.running:
            try:
                _socket.sendto(
                    EchoHeader.PING.value, (broadcast_ip, DISCOVERY_PORT)
                )
                address_bytes = await asyncio.wait_for(
                    loop.sock_recv(_socket, 2048),
                    timeout=time_out,
                )
                master_addr, _ = split_byte_to_str(address_bytes)
                master_address = tuple(loads(master_addr))
                self.connected = True
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Failed to broadcast the ping: {e}")
                traceback.print_exc()
            await async_sleep(0.1)
        return master_address

    async def heartbeat_loop(
        self,
        _socket: socket.socket,
        master_address: Address,
    ):
        # set up udp socket
        try:
            self.submit_task(
                self.update_info_loop, _socket, 2 * HEARTBEAT_INTERVAL
            )
        except Exception as e:
            logger.error(f"Failed to create udp socket: {e}")
            raise Exception("Failed to create udp socket")
        logger.info(
            f"The Net Manager starts broadcasting at {self.local_info['ip']}"
        )
        while self.connected:
            try:
                msg = generate_node_msg(self.local_info)
                # change broadcast ip to master ip
                _socket.sendto(msg, master_address)
                await async_sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
                traceback.print_exc()
        logger.info(
            "Heartbeat loop has been stopped, waiting for other master node"
        )

    async def update_info_loop(self, _socket: socket.socket, time_out: float):
        _socket.setblocking(False)
        # _socket.settimeout(1)  # should be used in blocking mode
        loop = asyncio.get_running_loop()
        while self.connected:
            try:
                msg = await asyncio.wait_for(
                    loop.sock_recv(_socket, 2048),
                    timeout=time_out,
                )
                assert type(loads(msg.decode())) == dict
                self.nodes_info_manager.update_nodes_info(
                    loads(msg.decode())
                )
                await async_sleep(0.1)
            except asyncio.TimeoutError:
                logger.warning("Timeout: The master node is offline")
                self.connected = False
            except Exception as e:
                logger.error(f"Error occurred in update info loop: {e}")
                traceback.print_exc()


class MasterNodeEchoUDPProtocol(asyncio.DatagramProtocol):

    def __init__(self, nodes_info_manager: MasterNodesInfoManager):
        self.nodes_info_manager = nodes_info_manager
        super().__init__()

    def connection_made(self, transport):
        self.transport = transport
        self.handler: Dict[bytes, Callable[[bytes, Address], bytes]] = {
            EchoHeader.PING.value: self.handle_ping,
            EchoHeader.HEARTBEAT.value: self.handle_heartbeat,
        }
        sock_name = transport.get_extra_info('sockname')
        logger.info(f"Master Node Echo UDP Server started at {sock_name}")
        self.addr: Address = (str(sock_name[0]), int(sock_name[1]))

    def datagram_received(self, data, addr):
        try:
            msg_type, msg_content = data[:1], data[1:]
            if msg_type not in self.handler:
                logger.error(f"Unknown Echo type: {msg_type}")
                return
            reply = self.handler[msg_type](msg_content, addr)
            self.transport.sendto(reply, addr)
        except Exception as e:
            logger.error(f"Error occurred in UDP received: {e}")
            traceback.print_exc()

    def handle_ping(self, content: bytes, addr: Address) -> bytes:
        return b"".join(
            [dumps(self.addr).encode(), b":", dumps(addr).encode()]
        )

    def handle_heartbeat(self, content: bytes, addr: Address) -> bytes:
        node_id, node_info = split_byte_to_str(content)
        self.nodes_info_manager.update_node(
            node_id, loads(node_info)
        )
        return self.nodes_info_manager.get_nodes_info_msg()


class MasterNodeManager(NodeManager):

    def __init__(self, host_ip: IPAddress, node_name: IPAddress) -> None:
        super().__init__(host_ip, node_name)
        # self.local_info["master"] = True

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        self.nodes_info_manager = MasterNodesInfoManager(self.local_info)
        asyncio.set_event_loop(self.loop)
        # self.submit_task(self.service_loop)
        self.submit_task(self.master_node_echo)
        self.loop.run_forever()

    async def master_node_echo(self):
        # Create the UDP server
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: MasterNodeEchoUDPProtocol(self.nodes_info_manager),
            local_addr=('0.0.0.0', DISCOVERY_PORT)  # Bind to all interfaces
        )
        logger.info("The Master starts to listen to broadcast")
        # check the heartbeat of nodes
        while self.running:
            remove_list = []
            for _id, last in self.nodes_info_manager.last_heartbeat.items():
                if time.time() - last > 3 * HEARTBEAT_INTERVAL:
                    remove_list.append(_id)
            for _id in remove_list:
                self.nodes_info_manager.remove_node(_id)
            await async_sleep(HEARTBEAT_INTERVAL)
        transport.close()
        self.stop_node()
        logger.info("Node discover task has been stopped")


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
