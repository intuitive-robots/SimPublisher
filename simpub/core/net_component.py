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

from .net_manager import NodeManagerBase
from .log import logger
from .utils import IPAddress, TopicName, ServiceName, HashIdentifier
from .utils import NodeInfo, INTERNAL_DISCOVER_PORT, EXTERNAL_DISCOVER_PORT
from .utils import MSG, NodeAddress, NetInfo, NetComponentInfo
from .utils import (
    split_byte,
    get_zmq_socket_port,
    create_address,
    generate_broadcast_msg,
)
from .utils import AsyncSocket


class NetComponent(abc.ABC):
    def __init__(self, name: str, with_local_namespace: bool):
        if NodeManagerBase.manager is None:
            raise ValueError("NetManager is not initialized")
        self.manager: NodeManagerBase = NodeManagerBase.manager
        self.running: bool = False
        self.host_ip: str = self.manager.local_info["addr"]["ip"]
        self.name = name
        if with_local_namespace:
            local_name: str = self.manager.local_info["name"]
            self.name = f"{local_name}/{name}"
        self.socket = self.create_socket()

    def shutdown(self) -> None:
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def create_socket(self) -> AsyncSocket:
        raise NotImplementedError

    @abc.abstractmethod
    def on_shutdown(self):
        self.running = False
        self.socket.close()


class Publisher(NetComponent):
    def __init__(self, topic_name: str, with_local_namespace: bool = False):
        super().__init__(topic_name, with_local_namespace)

    def create_socket(self) -> AsyncSocket:
        return self.manager.create_topic(self.name)

    def publish_bytes(self, data: bytes) -> None:
        msg = b"".join([f"{self.name}:".encode(), b"|", data])
        self.manager.submit_task(self.send_bytes_async, msg)

    def publish_dict(self, data: Dict) -> None:
        self.publish_string(dumps(data))

    def publish_string(self, string: str) -> None:
        msg = f"{self.name}:{string}"
        self.manager.submit_task(self.send_bytes_async, msg.encode())

    def on_shutdown(self) -> None:
        pass
        # self.manager.remove_local_topic(self.topic_name)

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
        self.topic_byte = self.name.encode("utf-8")
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
        logger.info(f"Topic {self.name} starts streaming")
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
                logger.error(f"Error when streaming {self.name}: {e}")
                traceback.print_exc()
        logger.info(f"Streamer for topic {self.name} is stopped")


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
        super().__init__(topic_name, False)
        self.connected = False
        self.callback = callback
        self.remote_addr: Optional[NodeAddress] = None

    def create_socket(self) -> AsyncSocket:
        sub_socket = self.manager.create_socket(zmq.SUB)
        return sub_socket

    def start_connection(self, addr: NodeAddress) -> None:
        """Changes the connection to a new IP address."""
        if self.connected and self.remote_addr is not None:
            logger.info(f"Disconnecting from {self.remote_addr}")
            self.socket.disconnect(
                f"tcp://{self.remote_addr['ip']}:{self.remote_addr['port']}"
            )
        self.socket.connect(f"tcp://{addr['ip']}:{addr['port']}")
        logger.info(f"Connected to {addr['ip']}:{addr['port']}")
        self.manager.submit_task(self.listen)
        self.remote_addr = addr
        self.connected = True

    async def wait_for_publisher(self) -> None:
        """Waits for a publisher to be available for the topic."""
        while self.running:
            node_info = self.manager.net_info_manager.check_topic(self.name)
            if node_info is not None:
                logger.info(
                    f"Connected to new publisher from node "
                    f"'{node_info['name']}' with topic '{self.name}'"
                )
                self.start_connection(node_info["addr"])
            await async_sleep(0.5)

    async def listen(self) -> None:
        """Listens for incoming messages on the subscribed topic."""
        while self.running:
            try:
                # Wait for a message
                msg = await self.socket.recv_string()
                self.callback(msg)
            except Exception:
                logger.error(
                    f"One error occurred in subscriber for topic '{self.name}"
                )
                traceback.print_exc()

    def on_shutdown(self) -> None:
        super().on_shutdown()


class AbstractService(NetComponent):

    def __init__(
        self,
        service_name: str,
    ) -> None:
        super().__init__(service_name, False)
        logger.info(f'"{self.name}" Service is ready')

    def create_socket(self) -> AsyncSocket:
        return self.manager.create_service(self.name)

    async def callback(self, msg: bytes):
        result = await asyncio.wait_for(
            self.manager.loop.run_in_executor(
                self.manager.executor, self.process_bytes_request, msg
            ),
            timeout=5.0,
        )
        await self.socket.send(result)

    @abc.abstractmethod
    def process_bytes_request(self, msg: bytes) -> bytes:
        raise NotImplementedError

    def on_shutdown(self):
        super().on_shutdown()
        logger.info(f'"{self.name}" Service is stopped')


class StrBytesService(AbstractService):

    def __init__(
        self,
        service_name: str,
        callback_func: Callable[[str], bytes],
    ) -> None:
        super().__init__(service_name)
        self.callback_func = callback_func

    def process_bytes_request(self, msg: bytes) -> bytes:
        return self.callback_func(msg.decode())


class StrService(AbstractService):

    def __init__(
        self,
        service_name: str,
        callback_func: Callable[[str], str],
    ) -> None:
        super().__init__(service_name)
        self.callback_func = callback_func

    def process_bytes_request(self, msg: bytes) -> bytes:
        return self.callback_func(msg.decode()).encode()
