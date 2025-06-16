from __future__ import annotations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import abc
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
from .utils import ClientNodeInfo, IPAddress, ServerNodeInfo, TopicName, ServiceName, HashIdentifier
from .utils import DISCOVERY_PORT
from .utils import MSG, NodeAddress
from .utils import split_byte, get_zmq_socket_port, create_address
from .utils import AsyncSocket


class NetComponent(abc.ABC):
    def __init__(self):
        if NodeManager.manager is None:
            raise ValueError("NodeManager is not initialized")
        self.manager: NodeManager = NodeManager.manager
        self.running: bool = False
        self.host_ip: str = self.manager.local_info["addr"]["ip"]
        self.local_name: str = self.manager.local_info["name"]

    def shutdown(self) -> None:
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(NetComponent):
    def __init__(self, topic_name: str, with_local_namespace: bool = False):
        super().__init__()
        self.topic_name = topic_name
        if with_local_namespace:
            self.topic_name = f"{self.local_name}/{topic_name}"
        self.socket = self.manager.pub_socket
        if self.manager.nodes_info_manager.check_topic(topic_name):
            logger.warning(f"Topic {topic_name} is already registered")
            raise ValueError(f"Topic {topic_name} is already registered")
        else:
            self.manager.register_local_topic(topic_name)
            logger.info(msg=f'Topic "{self.topic_name}" is ready to publish')

    def publish_bytes(self, data: bytes) -> None:
        msg = b''.join([f"{self.topic_name}".encode(), b"|", data])
        self.manager.submit_task(self.send_bytes_async, msg)

    def publish_dict(self, data: Dict) -> None:
        self.publish_string(dumps(data))

    def publish_string(self, string: str) -> None:
        msg = f"{self.topic_name}|{string}"
        self.manager.submit_task(self.send_bytes_async, msg.encode())

    def on_shutdown(self) -> None:
        self.manager.remove_local_topic(self.topic_name)

    async def send_bytes_async(self, msg: bytes) -> None:
        await self.socket.send(msg)


class Streamer(Publisher):
    def __init__(
        self,
        topic_name: str,
        update_func: Callable[[], Optional[Union[str, bytes, Dict]]],
        fps: int,
        start_streaming: bool = False,
    ):
        super().__init__(topic_name)
        self.running = False
        self.dt: float = 1 / fps
        self.update_func = update_func
        self.topic_byte = self.topic_name.encode("utf-8")
        if start_streaming:
            self.start_streaming()

    def start_streaming(self):
        self.manager.submit_task(self.update_loop)

    def generate_byte_msg(self) -> bytes:
        update_msg = self.update_func()
        if isinstance(update_msg, str):
            return update_msg.encode("utf-8")
        elif isinstance(update_msg, bytes):
            return update_msg
        elif isinstance(update_msg, dict):
            # return dumps(update_msg).encode("utf-8")
            return dumps(
                {
                    "updateData": self.update_func(),
                    "time": time.monotonic(),
                }
            ).encode("utf-8")
        raise ValueError("Update function should return str, bytes or dict")

    async def update_loop(self):
        self.running = True
        last = 0.0
        logger.info(f"Topic {self.topic_name} starts streaming")
        while self.running:
            try:
                diff = time.monotonic() - last
                if diff < self.dt:
                    await async_sleep(self.dt - diff)
                last = time.monotonic()
                await self.socket.send(
                    b"".join([self.topic_byte, b"|", self.generate_byte_msg()])
                )
            except Exception as e:
                logger.error(f"Error when streaming {self.topic_name}: {e}")
                traceback.print_exc()
        logger.info(f"Streamer for topic {self.topic_name} is stopped")


class ByteStreamer(Streamer):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], bytes],
        fps: int,
    ):
        super().__init__(topic, update_func, fps)
        self.update_func: Callable[[], bytes]

    def generate_byte_msg(self) -> bytes:
        return self.update_func()


class Subscriber(NetComponent):
    # TODO: test this class
    def __init__(self, topic_name: str, callback: Callable[[str], None]):
        super().__init__()
        self.sub_socket: AsyncSocket = self.manager.create_socket(zmq.SUB)
        self.topic_name = topic_name
        self.connected = False
        self.callback = callback
        self.remote_addr: Optional[NodeAddress] = None
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic_name)

    def change_connection(self, new_addr: NodeAddress) -> None:
        """Changes the connection to a new IP address."""
        if self.connected and self.remote_addr is not None:
            logger.info(f"Disconnecting from {self.remote_addr}")
            self.sub_socket.disconnect(
                f"tcp://{self.remote_addr}"
            )
        self.sub_socket.connect(f"tcp://{new_addr}")
        self.remote_addr = new_addr
        self.connected = True

    async def wait_for_publisher(self) -> None:
        """Waits for a publisher to be available for the topic."""
        while self.running:
            node_info = self.manager.nodes_info_manager.check_topic(
                self.topic_name
            )
            if node_info is not None:
                logger.info(
                    f"Connected to new publisher from node "
                    f"'{node_info['name']}' with topic '{self.topic_name}'"
                )
            await async_sleep(0.5)

    async def listen(self) -> None:
        """Listens for incoming messages on the subscribed topic."""
        while self.running:
            try:
                # Wait for a message
                msg = await self.sub_socket.recv_string()
                # Invoke the callback
                self.callback(msg)
            except Exception as e:
                logger.error(
                    f"Error in subscriber for topic '{self.topic_name}': {e}"
                )
                traceback.print_exc()

    def on_shutdown(self) -> None:
        self.running = False
        self.sub_socket.close()

# region SERVER_CODE

# class AbstractService(NetComponent):

#     def __init__(
#         self,
#         service_name: str,
#     ) -> None:
#         super().__init__()
#         self.service_name = service_name
#         self.socket = self.manager.service_socket
#         # register service
#         self.manager.local_info["serviceList"].append(service_name)
#         self.manager.service_cbs[service_name.encode()] = self.callback
#         logger.info(f'"{self.service_name}" Service is ready')

#     async def callback(self, msg: bytes):
#         result = await asyncio.wait_for(
#             self.manager.loop.run_in_executor(
#                 self.manager.executor, self.process_bytes_request, msg
#             ),
#             timeout=5.0,
#         )
#         await self.socket.send(result)

#     @abc.abstractmethod
#     def process_bytes_request(self, msg: bytes) -> bytes:
#         raise NotImplementedError

#     def on_shutdown(self):
#         self.manager.local_info["serviceList"].remove(self.service_name)
#         logger.info(f'"{self.service_name}" Service is stopped')


# class StrBytesService(AbstractService):

#     def __init__(
#         self,
#         service_name: str,
#         callback_func: Callable[[str], bytes],
#     ) -> None:
#         super().__init__(service_name)
#         self.callback_func = callback_func

#     def process_bytes_request(self, msg: bytes) -> bytes:
#         return self.callback_func(msg.decode())


# class StrService(AbstractService):

#     def __init__(
#         self,
#         service_name: str,
#         callback_func: Callable[[str], str],
#     ) -> None:
#         super().__init__(service_name)
#         self.callback_func = callback_func

#     def process_bytes_request(self, msg: bytes) -> bytes:
#        return self.callback_func(msg.decode()).encode()

# endregion

class NodeInfoManager:

    def __init__(self, local_info: ClientNodeInfo) -> None:
        self.server_nodes_info: Dict[HashIdentifier, ServerNodeInfo] = {}
        # local info is the master node info
        self.local_info = local_info
        self.node_id = local_info["nodeID"]

    
    def get_server_nodes_info(self) -> Dict[HashIdentifier, ServerNodeInfo]:
        return self.server_nodes_info

    def check_service(self, service_name: ServiceName) -> Optional[ServerNodeInfo]:
        for info in self.server_nodes_info.values():
            if service_name in info["serviceList"]:
                return info
        return None

    def check_topic(self, topic_name: TopicName) -> Optional[ServerNodeInfo]:
        for info in self.server_nodes_info.values():
            if topic_name in info["topicList"]:
                return info
        return None

    def register_server_node(self, info: ServerNodeInfo):
        node_id = info["nodeID"]
        if node_id not in self.server_nodes_info.keys():
            logger.info(
                f"Node {info['name']} from "
                f"{info['addr']['ip']} has been launched"
            )
        self.server_nodes_info[node_id] = info

    def remove_server_node(self, node_id: HashIdentifier):
        try:
            if node_id in self.server_nodes_info.keys():
                removed_info = self.server_nodes_info.pop(node_id)
                logger.info(f"Node {removed_info['name']} is offline")
        except Exception as e:
            logger.error(f"Error occurred when removing node: {e}")

    def get_node_info(self, node_name: str) -> Optional[ServerNodeInfo]:
        for info in self.server_nodes_info.values():
            if info["name"] == node_name:
                return info
        return None


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
        self.local_info: ClientNodeInfo = {
            "name": node_name,
            "nodeID": str(uuid.uuid4()),
            "addr": create_address(host_ip, DISCOVERY_PORT),
            "type": "Master",
            # "servicePort": get_zmq_socket_port(self.service_socket),
            "topicPort": get_zmq_socket_port(self.pub_socket),
            # "serviceList": [],
            "topicList": [],
        }
        logger.info(f"Node {node_name} starts at {host_ip}:{DISCOVERY_PORT}")
        self.nodes_info_manager = NodeInfoManager(self.local_info)
        # start the server in a thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.thread_task)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)
        logger.info(f"Node {self.local_info['name']} is initialized")

    def start_node_broadcast(self):
        self.submit_task(self.broadcast_loop)

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

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

    def stop_node(self):
        logger.info("Start to stop the node")
        self.running = False
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.stop_tasks)
                time.sleep(0.1)
                self.loop.stop()
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        logger.info("Start to shutdown the executor")
        self.executor.shutdown(wait=False)
        logger.info("The executor has been shutdown")
        NodeManager.manager = None

    def stop_tasks(self):
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            if task is asyncio.current_task():
                continue
            task.cancel()
        logger.info("All tasks have been cancelled")

    def spin(self):
        while True:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        self.stop_node()
        logger.info("The node has been stopped")

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    # async def service_loop(self):
    #     logger.info("The service loop is running...")
    #     service_socket = self.service_socket
    #     while self.running:
    #         bytes_msg = await service_socket.recv_multipart()
    #         service_name, request = split_byte(b"".join(bytes_msg))
    #         # the zmq service socket is blocked and only run one at a time
    #         if service_name in self.service_cbs.keys():
    #             try:
    #                 await self.service_cbs[service_name](request)
    #             except asyncio.TimeoutError:
    #                 logger.error("Timeout: callback function took too long")
    #                 await service_socket.send(MSG.SERVICE_TIMEOUT.value)
    #             except Exception as e:
    #                 logger.error(
    #                     f"One error occurred when processing the Service "
    #                     f'"{service_name}": {e}'
    #                 )
    #                 traceback.print_exc()
    #                 await service_socket.send(MSG.SERVICE_ERROR.value)
    #         await async_sleep(0.01)
    #     logger.info("Service loop has been stopped")

    def start_event_loop(self):
        # TODO: remove services

        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        # self.submit_task(self.service_loop)
        # self.register_service = StrService(
        #     "RegisterNode",
        #     self.register_node_callback,
        # )
        # self.node_offline_service = StrService(
        #     "NodeOffline",
        #     self.node_offline_callback,
        # )
        # self.get_nodes_info_service = StrService(
        #     "GetNodesInfo",
        #     self.get_nodes_info_callback,
        # )
        self.loop.run_forever()

    async def broadcast_loop(self):

        # TODO: rewrite to receive the local info from the server
        # also use self.nodes_info_manager.register_server_node(server_info)

        logger.info("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # calculate broadcast ip
        local_info = self.local_info
        _ip = local_info["addr"]["ip"]
        ip_bin = struct.unpack('!I', socket.inet_aton(_ip))[0]
        netmask_bin = struct.unpack('!I', socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack('!I', broadcast_bin))
        while self.running:
            msg = f"SimPub|{dumps(local_info)}"
            _socket.sendto(
                msg.encode(), (broadcast_ip, DISCOVERY_PORT)
            )
            await async_sleep(0.1)
        logger.info("Broadcasting has been stopped")

    # def register_node_callback(self, msg: str) -> str:
    #     # NOTE: something wrong with sending message, but it solved somehow
    #     client_info: NodeInfo = loads(msg)
    #     # NOTE: the client info may be updated so the reference cannot be used
    #     # NOTE: TypeDict is somehow block if the key is not in the dict
    #     self.nodes_info_manager.register_node(client_info)
    #     return "The info has been registered"

    def node_offline_callback(self, msg: str) -> str:
        client_name = msg
        self.nodes_info_manager.remove_server_node(client_name)
        return "The info has been removed"

    def get_server_nodes_info_callback(
        self,
        msg: str,
    ) -> str:
        return dumps(self.nodes_info_manager.get_server_nodes_info())

    # def register_local_service(self, service_name: ServiceName) -> None:
    #     if self.nodes_info_manager.check_service(service_name) is not None:
    #         logger.warning(
    #             f"Service {service_name} has been registered, "
    #             f"cannot register again"
    #         )
    #         raise RuntimeError("Service has been registered")
    #     if service_name in self.local_info["serviceList"]:
    #         raise RuntimeError("Service has been registered")
    #     self.local_info["serviceList"].append(service_name)

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

    # def remove_local_service(self, service_name: ServiceName) -> None:
    #     if service_name in self.local_info["serviceList"]:
    #         self.local_info["serviceList"].remove(service_name)

    def remove_local_topic(self, topic_name: TopicName) -> None:
        if topic_name in self.local_info["topicList"]:
            self.local_info["topicList"].remove(topic_name)


def init_node(
    ip_addr: str,
    node_name: str,
) -> NodeManager:
    if NodeManager.manager is not None:
        raise RuntimeError("The node has been initialized")
    return NodeManager(ip_addr, node_name)