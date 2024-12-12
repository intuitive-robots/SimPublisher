import abc
import asyncio
from asyncio import sleep as async_sleep
from typing import Callable, Dict, List, Optional, Union
from json import dumps
import time
import zmq
import traceback

from .net_manager import NodeManager
from .log import logger
from .utils import AsyncSocket, Address


class NetComponent(abc.ABC):
    def __init__(self):
        if NodeManager.manager is None:
            raise ValueError("NodeManager is not initialized")
        self.manager: NodeManager = NodeManager.manager
        self.node_info_manager = self.manager.nodes_info_manager
        self.running: bool = False
        self.host_ip: str = self.manager.local_info["ip"]
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
        if self.node_info_manager.check_topic(topic_name):
            logger.warning(f"Topic {topic_name} is already registered")
            raise ValueError(f"Topic {topic_name} is already registered")
        else:
            self.node_info_manager.register_topic(topic_name)
            logger.info(msg=f'Topic "{self.topic_name}" is ready to publish')

    def publish_bytes(self, data: bytes) -> None:
        msg = b''.join([f"{self.topic_name}:".encode(), b":", data])
        self.manager.submit_task(self.send_bytes_async, msg)

    def publish_dict(self, data: Dict) -> None:
        self.publish_string(dumps(data))

    def publish_string(self, string: str) -> None:
        msg = f"{self.topic_name}:{string}"
        self.manager.submit_task(self.send_bytes_async, msg.encode())

    def on_shutdown(self) -> None:
        self.node_info_manager.remove_local_topic(self.topic_name)

    async def send_bytes_async(self, msg: bytes) -> None:
        await self.socket.send(msg)


class Streamer(Publisher):
    def __init__(
        self,
        topic_name: str,
        update_func: Callable[[], Optional[Union[str, bytes, Dict]]],
        fps: int = 45,
        start_streaming: bool = True,
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
        return dumps(
            {
                "updateData": self.update_func(),
                "time": time.monotonic(),
            }
        ).encode("utf-8")

    async def update_loop(self):
        self.running = True
        last = 0.0
        while self.running:
            try:
                diff = time.monotonic() - last
                if diff < self.dt:
                    await async_sleep(self.dt - diff)
                last = time.monotonic()
                await self.socket.send(
                    b"".join([self.topic_byte, b":", self.generate_byte_msg()])
                )
            except Exception as e:
                logger.error(f"Error when streaming {self.topic_name}: {e}")
        logger.info(f"Streamer for topic {self.topic_name} is stopped")


class ByteStreamer(Streamer):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], bytes],
        fps: int = 45,
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
        self.remote_addr: Optional[Address] = None
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic_name)

    def change_connection(self, new_addr: Address) -> None:
        """Changes the connection to a new IP address."""
        if self.connected and self.remote_addr is not None:
            logger.info(f"Disconnecting from {self.remote_addr}")
            self.sub_socket.disconnect(
                f"tcp://{self.remote_addr[0]}:{self.remote_addr[1]}"
            )
        self.sub_socket.connect(f"tcp://{new_addr[0]}:{new_addr[1]}")
        self.remote_addr = new_addr
        self.connected = True

    async def wait_for_publisher(self) -> None:
        """Waits for a publisher to be available for the topic."""
        while self.running:
            addr = self.manager.nodes_info_manager.check_topic(self.topic_name)
            if addr is not None and addr != self.remote_addr:
                self.change_connection(addr)
                logger.info(
                    f"Connected to new publisher at {addr} for topic "
                    f"'{self.topic_name}'"
                )
            await async_sleep(0.5)

    async def listen(self) -> None:
        """Listens for incoming messages on the subscribed topic."""
        while self.running:
            try:
                # Wait for a message
                msg = await self.sub_socket.recv_string()
                self.callback(msg)
                # Invoke the callback
                self.callback(msg)
            except Exception as e:
                logger.error(
                    f"Error in subscriber for topic '{self.topic_name}': {e}"
                )
                traceback.print_exc()


class Service(NetComponent):
    def __init__(
        self,
        service_name: str,
        callback: Callable[[str], Union[str, bytes, Dict]],
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.callback_func = callback
        self.socket = self.manager.service_socket
        # register service
        self.manager.local_info["serviceList"].append(service_name)
        self.on_trigger_events: List[Callable[[str], None]] = []
        logger.info(f'"{self.service_name}" Service is ready')

    async def callback(self, msg: str):
        result = await asyncio.wait_for(
            self.manager.loop.run_in_executor(
                self.manager.executor, self.callback_func, msg
            ),
            timeout=5.0,
        )
        for event in self.on_trigger_events:
            event(msg)
        await self.send(result)

    @abc.abstractmethod
    async def send(self, data: Union[str, bytes, Dict]):
        raise NotImplementedError

    def on_shutdown(self):
        self.manager.local_info["serviceList"].remove(self.service_name)
        logger.info(f'"{self.service_name}" Service is stopped')


class StringService(Service):
    async def send(self, string: str):
        await self.socket.send_string(string)


class BytesService(Service):
    async def send(self, data: bytes):
        await self.socket.send(data)


class DictService(Service):
    async def send(self, data: Dict):
        await self.socket.send_string(dumps(data))
