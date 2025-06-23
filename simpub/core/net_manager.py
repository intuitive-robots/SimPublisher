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
from .utils import MCAST_GRP, XRNodeInfo, IPAddress, XRNodeInfo, TopicName, ServiceName, HashIdentifier
from .utils import DISCOVERY_PORT
from .utils import MSG, NodeAddress
from .utils import split_byte, get_zmq_socket_port, create_address
from .utils import send_string_request


# class NetComponent(abc.ABC):
#     def __init__(self):
#         if XRNodeManager.manager is None:
#             raise ValueError("XRNodeManager is not initialized")
#         self.manager: XRNodeManager = XRNodeManager.manager
#         self.running: bool = False
#         self.host_ip: str = self.manager.local_info["addr"]["ip"]
#         self.local_name: str = self.manager.local_info["name"]

#     def shutdown(self) -> None:
#         self.running = False
#         self.on_shutdown()

#     @abc.abstractmethod
#     def on_shutdown(self):
#         raise NotImplementedError


# class Publisher(NetComponent):
#     def __init__(self, topic_name: str, with_local_namespace: bool = False):
#         super().__init__()
#         self.topic_name = topic_name
#         if with_local_namespace:
#             self.topic_name = f"{self.local_name}/{topic_name}"
#         self.socket = self.manager.pub_socket
#         if self.manager.nodes_info_manager.check_topic(topic_name):
#             logger.warning(f"Topic {topic_name} is already registered")
#             raise ValueError(f"Topic {topic_name} is already registered")
#         else:
#             self.manager.register_local_topic(topic_name)
#             logger.info(msg=f'Topic "{self.topic_name}" is ready to publish')

#     def publish_bytes(self, data: bytes) -> None:
#         msg = b''.join([f"{self.topic_name}".encode(), b"|", data])
#         self.manager.submit_task(self.send_bytes_async, msg)

#     def publish_dict(self, data: Dict) -> None:
#         self.publish_string(dumps(data))

#     def publish_string(self, string: str) -> None:
#         msg = f"{self.topic_name}|{string}"
#         self.manager.submit_task(self.send_bytes_async, msg.encode())

#     def on_shutdown(self) -> None:
#         self.manager.remove_local_topic(self.topic_name)

#     async def send_bytes_async(self, msg: bytes) -> None:
#         await self.socket.send(msg)


# class Streamer(Publisher):
#     def __init__(
#         self,
#         topic_name: str,
#         update_func: Callable[[], Optional[Union[str, bytes, Dict]]],
#         fps: int,
#         start_streaming: bool = False,
#     ):
#         super().__init__(topic_name)
#         self.running = False
#         self.dt: float = 1 / fps
#         self.update_func = update_func
#         self.topic_byte = self.topic_name.encode("utf-8")
#         if start_streaming:
#             self.start_streaming()

#     def start_streaming(self):
#         self.manager.submit_task(self.update_loop)

#     def generate_byte_msg(self) -> bytes:
#         update_msg = self.update_func()
#         if isinstance(update_msg, str):
#             return update_msg.encode("utf-8")
#         elif isinstance(update_msg, bytes):
#             return update_msg
#         elif isinstance(update_msg, dict):
#             # return dumps(update_msg).encode("utf-8")
#             return dumps(
#                 {
#                     "updateData": self.update_func(),
#                     "time": time.monotonic(),
#                 }
#             ).encode("utf-8")
#         raise ValueError("Update function should return str, bytes or dict")

#     async def update_loop(self):
#         self.running = True
#         last = 0.0
#         logger.info(f"Topic {self.topic_name} starts streaming")
#         while self.running:
#             try:
#                 diff = time.monotonic() - last
#                 if diff < self.dt:
#                     await async_sleep(self.dt - diff)
#                 last = time.monotonic()
#                 await self.socket.send(
#                     b"".join([self.topic_byte, b"|", self.generate_byte_msg()])
#                 )
#             except Exception as e:
#                 logger.error(f"Error when streaming {self.topic_name}: {e}")
#                 traceback.print_exc()
#         logger.info(f"Streamer for topic {self.topic_name} is stopped")


# class ByteStreamer(Streamer):
#     def __init__(
#         self,
#         topic: str,
#         update_func: Callable[[], bytes],
#         fps: int,
#     ):
#         super().__init__(topic, update_func, fps)
#         self.update_func: Callable[[], bytes]

#     def generate_byte_msg(self) -> bytes:
#         return self.update_func()


# class Subscriber(NetComponent):
#     # TODO: test this class
#     def __init__(self, topic_name: str, callback: Callable[[str], None]):
#         super().__init__()
#         self.sub_socket: AsyncSocket = self.manager.create_socket(zmq.SUB)
#         self.topic_name = topic_name
#         self.connected = False
#         self.callback = callback
#         self.remote_addr: Optional[NodeAddress] = None
#         self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic_name)

#     def change_connection(self, new_addr: NodeAddress) -> None:
#         """Changes the connection to a new IP address."""
#         if self.connected and self.remote_addr is not None:
#             logger.info(f"Disconnecting from {self.remote_addr}")
#             self.sub_socket.disconnect(
#                 f"tcp://{self.remote_addr}"
#             )
#         self.sub_socket.connect(f"tcp://{new_addr}")
#         self.remote_addr = new_addr
#         self.connected = True

#     async def wait_for_publisher(self) -> None:
#         """Waits for a publisher to be available for the topic."""
#         while self.running:
#             node_info = self.manager.nodes_info_manager.check_topic(
#                 self.topic_name
#             )
#             if node_info is not None:
#                 logger.info(
#                     f"Connected to new publisher from node "
#                     f"'{node_info['name']}' with topic '{self.topic_name}'"
#                 )
#             await async_sleep(0.5)

#     async def listen(self) -> None:
#         """Listens for incoming messages on the subscribed topic."""
#         while self.running:
#             try:
#                 # Wait for a message
#                 msg = await self.sub_socket.recv_string()
#                 # Invoke the callback
#                 self.callback(msg)
#             except Exception as e:
#                 logger.error(
#                     f"Error in subscriber for topic '{self.topic_name}': {e}"
#                 )
#                 traceback.print_exc()

#     def on_shutdown(self) -> None:
#         self.running = False
#         self.sub_socket.close()



class XRNodeManager:

    manager = None

    def __init__(self, host_ip: IPAddress) -> None:
        XRNodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context()  # type: ignore
        # # publisher
        # self.pub_socket = self.create_socket(zmq.PUB)
        # self.pub_socket.bind(f"tcp://{host_ip}:0")
        # # service
        # self.service_socket = self.zmq_context.socket(zmq.REP)
        # self.service_socket.bind(f"tcp://{host_ip}:0")
        # self.service_cbs: Dict[bytes, Callable[[bytes], Awaitable]] = {}
        self.xr_nodes_info: Dict[HashIdentifier, XRNodeInfo] = {}
        # self.nodes_info_manager = NodeInfoManager(self.local_info)
        # start the server in a thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.thread_task)
        # wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)
        # logger.info(f"Node {self.local_info['name']} is initialized")

    def start_discover_node_loop(self):
        self.executor.submit(self.xr_node_discover_loop)

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
        XRNodeManager.manager = None

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

    def submit_asyncio_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def start_event_loop(self):
        # TODO: remove services

        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def xr_node_discover_loop(self):
        # Define multicast group details
        multicast_group = '239.255.10.10'
        port = 7720
        # Create UDP socket
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
            socket.IPPROTO_UDP
        )
        # Allow reuse of address
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the port
        sock.bind(('', port))
        # Tell the operating system to add the socket to the multicast group
        # on all interfaces
        group = socket.inet_aton(multicast_group)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print(f"Listening for multicast messages on {multicast_group}:{port}")
        try:
            while True:
                # Receive data from the socket (2048 is buffer size)
                data, address = sock.recvfrom(2048)
                # Filter by sender IP address
                node_ip = address[0]
                # Decode and print the message from the allowed IP
                message = data.decode('utf-8')
                node_id, node_port = message[:36], message[36:]
                if node_id not in self.xr_nodes_info:
                    node_address = f"tcp://{node_ip}:{node_port}"
                    self.register_node_info(node_id, node_address)
        except KeyboardInterrupt:
            print("Stopping multicast receiver...")
        finally:
            # Close the socket when done
            sock.close()

    def register_node_info(self, node_id: str, node_address: str) -> None:
        """
        Register a new XR node info.
        If the node info already exists, it will be updated.
        """
        node_info_bytes = send_string_request("GetNodeInfo", "", node_address)
        self.xr_nodes_info[node_id] = loads(node_info_bytes.decode('utf-8'))
        print(f"Registering node info: {node_id} at {node_address}")


def init_xr_node_manager(ip_addr: str) -> XRNodeManager:
    if XRNodeManager.manager is not None:
        raise RuntimeError("The node has been initialized")
    return XRNodeManager(ip_addr)
