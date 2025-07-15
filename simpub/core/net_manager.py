from __future__ import annotations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import abc
from typing import Dict, Optional, Callable, Union
import asyncio
from asyncio import sleep as async_sleep
import socket
import struct
import zmq
import zmq.asyncio
import time
from json import dumps, loads
import traceback

from .log import logger
from .utils import (
    send_string_request_async,
    MULTICAST_GRP,
    DISCOVERY_PORT,
    XRNodeInfo,
)


class NetComponent(abc.ABC):
    def __init__(self):
        if XRNodeManager.manager is None:
            raise ValueError("XRNodeManager is not initialized")
        self.manager: XRNodeManager = XRNodeManager.manager
        self.running: bool = False

    def shutdown(self) -> None:
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(NetComponent):
    def __init__(self, topic_name: str):
        super().__init__()
        self.topic_name = topic_name
        self.socket = self.manager.create_socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.manager.host_ip}:0")

    def publish_bytes(self, msg: bytes) -> None:
        self.manager.submit_asyncio_task(self.send_bytes_async, msg)

    def publish_dict(self, msg: Dict) -> None:
        self.publish_string(dumps(msg))

    def publish_string(self, msg: str) -> None:
        self.manager.submit_asyncio_task(self.send_bytes_async, msg.encode())

    async def send_bytes_async(self, msg: bytes) -> None:
        await self.socket.send(msg)

    def on_shutdown(self):
        pass


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
        self.manager.submit_asyncio_task(self.update_loop)

    def generate_byte_msg(self) -> bytes:
        update_msg = self.update_func()
        if isinstance(update_msg, str):
            return update_msg.encode("utf-8")
        elif isinstance(update_msg, bytes):
            return update_msg
        elif isinstance(update_msg, dict):
            return dumps(update_msg).encode("utf-8")
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
                await self.send_bytes_async(self.generate_byte_msg())
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

# NOTE: asyncio.loop.sock_recvfrom can only be used after Python 3.11
# So we create a custom DatagramProtocol for multicast discovery
class MulticastDiscoveryProtocol(asyncio.DatagramProtocol):
    """DatagramProtocol for handling multicast discovery messages"""

    def __init__(self, node_manager: XRNodeManager):
        self.node_manager = node_manager
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        logger.info("Multicast discovery connection established")

    def datagram_received(self, data, addr):
        """Handle incoming multicast discovery messages"""
        try:
            # Filter by sender IP address
            node_ip = addr[0]
            # Decode and print the message from the allowed IP
            message = data.decode("utf-8")
            node_id, service_port = message[:36], message[36:]
            if node_id not in self.node_manager.xr_nodes_info:
                # Schedule the async registration
                self.node_manager.submit_asyncio_task(
                    self.node_manager.async_register_node_info,
                    node_id,
                    node_ip,
                    service_port,
                )

        except Exception as e:
            logger.error(f"Error processing datagram: {e}")
            traceback.print_exc()

    def error_received(self, exc):
        logger.error(f"Multicast protocol error: {exc}")

    def connection_lost(self, exc):
        if exc:
            logger.error(f"Multicast connection lost: {exc}")
        else:
            logger.error("Multicast discovery connection closed")


class XRNodeManager:
    manager = None

    def __init__(self, host_ip: str) -> None:
        XRNodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context.instance()
        self.host_ip: str = host_ip
        self.xr_nodes_info: Dict[str, XRNodeInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.server_future = self.executor.submit(self.thread_task)
        self.discovery_task = None  # Track the discovery task
        self.discovery_transport = None  # Track the transport for cleanup
        # Wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)

    def start_discover_node_loop(self):
        """Start the async discovery loop in the event loop"""
        if self.loop and self.loop.is_running():
            # Submit the async task to the event loop
            self.discovery_task = self.submit_asyncio_task(
                self.xr_node_discover_loop
            )
        else:
            logger.error(
                "Event loop is not running, cannot start discovery loop"
            )

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
        # Close discovery transport if it exists
        if self.discovery_transport:
            self.discovery_transport.close()

        # Cancel all running tasks including discovery
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
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def xr_node_discover_loop(self):
        # Create multicast socket
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        # Allow reuse of address
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the port
        sock.bind(("", DISCOVERY_PORT))
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(MULTICAST_GRP),
            socket.inet_aton(self.host_ip),
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        logger.info(
            f"Listening for multicast on {MULTICAST_GRP}:{DISCOVERY_PORT}"
        )
        # Get event loop and create datagram endpoint
        loop = asyncio.get_event_loop()

        try:
            # Create the datagram endpoint with the protocol
            transport, _ = await loop.create_datagram_endpoint(
                lambda: MulticastDiscoveryProtocol(self), sock=sock
            )
            # Store transport for cleanup
            self.discovery_transport = transport
            # Keep the loop running until stopped
            while self.running:
                await asyncio.sleep(1)  # Keep the coroutine alive
        except asyncio.CancelledError:
            logger.info("Discovery loop cancelled...")
        except Exception as e:
            logger.error(f"Error in discovery loop: {e}")
            traceback.print_exc()
        finally:
            # Clean up
            if "transport" in locals():
                transport.close()
            sock.close()
            logger.info("Multicast discovery loop stopped")

    async def async_register_node_info(
        self, node_id: str, node_ip: str, service_port: str
    ) -> None:
        try:
            node_info_bytes = await send_string_request_async(
                ["GetNodeInfo", ""], f"tcp://{node_ip}:{service_port}"
            )
            if node_info_bytes is None:
                logger.error(
                    f"Failed to get node info from {node_ip}:{service_port}"
                )
                return
            self.xr_nodes_info[node_id] = loads(
                node_info_bytes.decode("utf-8")
            )
            self.xr_nodes_info[node_id]["ip"] = node_ip
            logger.info(
                f"Registering node info: {node_id} at {node_ip}:{service_port}"
            )
        except Exception as e:
            logger.error(f"Error in async_register_node_info: {e}")
            traceback.print_exc()


def init_xr_node_manager(ip_addr: str) -> XRNodeManager:
    if XRNodeManager.manager is not None:
        raise RuntimeError("The node has been initialized")
    return XRNodeManager(ip_addr)
