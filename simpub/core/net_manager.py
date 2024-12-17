from __future__ import annotations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Awaitable
import asyncio
from asyncio import sleep as async_sleep
import zmq
import zmq.asyncio
import time
import uuid
from json import dumps, loads
import traceback

from .log import logger
from .utils import IPAddress, TopicName, ServiceName, HashIdentifier
from .utils import NodeInfo, DISCOVERY_PORT, HEARTBEAT_INTERVAL
from .utils import EchoHeader, MSG, NodeAddress, NodeTypes
from .utils import split_byte, get_zmq_socket_port, create_address


class NodesInfoManager:

    def __init__(self, local_info: NodeInfo) -> None:
        self.nodes_info: Dict[HashIdentifier, NodeInfo] = {}
        # local info is the master node info
        self.local_info = local_info
        self.node_id = local_info["nodeID"]
        self.last_heartbeat: Dict[HashIdentifier, float] = {}

    def get_nodes_info_msg(self) -> bytes:
        return dumps(self.nodes_info).encode()

    def get_local_info_msg(self) -> bytes:
        return dumps(self.local_info).encode()

    def check_service(self, service_name: ServiceName) -> Optional[NodeInfo]:
        for info in self.nodes_info.values():
            if service_name in info["serviceList"]:
                return info
        return None

    def check_topic(self, topic_name: TopicName) -> Optional[NodeInfo]:
        for info in self.nodes_info.values():
            if topic_name in info["topicList"]:
                return info
        return None

    def update_node(self, info: NodeInfo):
        node_id = info["nodeID"]
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


class MasterEchoUDPProtocol(asyncio.DatagramProtocol):

    def __init__(self, nodes_info_manager: NodesInfoManager):
        self.nodes_info_manager = nodes_info_manager
        super().__init__()

    def connection_made(self, transport):
        self.transport = transport
        self.handler: Dict[bytes, Callable[[bytes, NodeAddress], bytes]] = {
            EchoHeader.PING.value: self.handle_ping,
            EchoHeader.HEARTBEAT.value: self.handle_heartbeat,
            EchoHeader.NODES.value: self.handle_nodes,
        }
        self.addr: NodeAddress = self.nodes_info_manager.local_info["addr"]
        logger.info(
            msg=f"Master Node Echo UDP Server started at "
            f"{self.addr['ip']}:{self.addr['port']}"
        )

    def datagram_received(self, data, addr):
        try:
            ping = data[:1]
            if ping not in self.handler:
                logger.error(f"Unknown Echo type: {ping}")
                return
            reply = self.handler[ping](data, create_address(*addr))
            self.transport.sendto(reply, addr)
        except Exception as e:
            logger.error(f"Error occurred in UDP received: {e}")
            traceback.print_exc()

    def handle_ping(self, data: bytes, addr: NodeAddress) -> bytes:
        return b"".join(
            [
                self.nodes_info_manager.get_local_info_msg(),
                b"|",
                dumps(addr).encode()
            ]
        )

    def handle_heartbeat(self, data: bytes, addr: NodeAddress) -> bytes:
        self.nodes_info_manager.update_node(loads(data[1:].decode()))
        return self.nodes_info_manager.get_local_info_msg()

    def handle_nodes(self, data: bytes, addr: NodeAddress) -> bytes:
        return b"".join(
            [
                self.nodes_info_manager.get_local_info_msg(),
                b"|",
                self.nodes_info_manager.get_nodes_info_msg()
            ]
        )


class NodeManager:

    manager = None

    def __init__(self, host_ip: IPAddress, node_name: IPAddress) -> None:
        NodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context()  # type: ignore
        # publisher
        self.pub_socket = self.create_socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:0")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:0")
        self.service_cbs: Dict[bytes, Callable[[bytes], Awaitable]] = {}
        # message for broadcasting
        self.local_info: NodeInfo = {
            "name": node_name,
            "nodeID": str(uuid.uuid4()),
            "addr": create_address(host_ip, DISCOVERY_PORT),
            "type": NodeTypes.MASTER.value,
            "servicePort": get_zmq_socket_port(self.service_socket),
            "topicPort": get_zmq_socket_port(self.pub_socket),
            "serviceList": [],
            "topicList": [],
        }
        self.nodes_info_manager = NodesInfoManager(self.local_info)
        print("length of local_info: ", len(dumps(self.local_info).encode()))
        # start the server in a thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.thread_task)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)
        logger.info(f"Node {self.local_info['name']} is initialized")

    def start_node_discover(self):
        self.submit_task(self.master_node_echo)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def thread_task(self):
        try:
            self.start_event_loop()
        except KeyboardInterrupt:
            self.stop_node()
        finally:
            logger.info("The node has been stopped")

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

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

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

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.submit_task(self.service_loop)
        self.loop.run_forever()

    async def master_node_echo(self):
        # Create the UDP server
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: MasterEchoUDPProtocol(self.nodes_info_manager),
            local_addr=("0.0.0.0", DISCOVERY_PORT),
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

    def register_local_service(self, service_name: ServiceName) -> None:
        if self.nodes_info_manager.check_service(service_name) is not None:
            logger.warning(
                f"Service {service_name} has been registered, "
                f"cannot register again"
            )
            raise RuntimeError("Service has been registered")
        if service_name in self.local_info["serviceList"]:
            raise RuntimeError("Service has been registered")
        self.local_info["serviceList"].append(service_name)

    def register_local_topic(self, topic_name: TopicName) -> None:
        if self.nodes_info_manager.check_topic(topic_name) is not None:
            logger.warning(
                f"Topic {topic_name} has been registered, "
                f"cannot register again"
            )
            raise RuntimeError("Topic has been registered")
        if topic_name in self.local_info["topicList"]:
            raise RuntimeError("Topic has been registered")
        self.local_info["topicList"].append(topic_name)

    def remove_local_service(self, service_name: ServiceName) -> None:
        if service_name in self.local_info["serviceList"]:
            self.local_info["serviceList"].remove(service_name)

    def remove_local_topic(self, topic_name: TopicName) -> None:
        if topic_name in self.local_info["topicList"]:
            self.local_info["topicList"].remove(topic_name)


def init_node(
    ip_addr: str,
    node_name: str = "PythonNode",
) -> NodeManager:
    if NodeManager.manager is not None:
        raise RuntimeError("The node has been initialized")
    return NodeManager(ip_addr, node_name)
