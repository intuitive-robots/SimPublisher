import abc
import asyncio
from asyncio import sleep as async_sleep
import time
from typing import Callable, Dict, List, Optional, Union
from json import dumps

from .net_manager import NetManager
from .log import logger


class NetComponent(abc.ABC):
    def __init__(self):
        if NetManager.manager is None:
            raise ValueError("NetManager is not initialized")
        self.manager: NetManager = NetManager.manager
        self.running: bool = False
        self.host_ip: str = self.manager.local_info["ip"]

    def shutdown(self) -> None:
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(NetComponent):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.socket = self.manager.pub_socket
        if topic in self.manager.local_info["topicList"]:
            logger.warning(f"Host {topic} is already registered")
        else:
            self.manager.local_info["topicList"].append(topic)
            logger.info(f'Publisher for topic "{self.topic}" is ready')

    def publish_bytes(self, data: bytes) -> None:
        msg = b''.join([f"{self.topic}:".encode(), b":", data])
        self.manager.submit_task(self.send_bytes_async, msg)

    def publish_dict(self, data: Dict) -> None:
        self.publish_string(dumps(data))

    def publish_string(self, string: str) -> None:
        msg = f"{self.topic}:{string}"
        self.manager.submit_task(self.send_bytes_async, msg.encode())

    def on_shutdown(self) -> None:
        self.manager.local_info["topicList"].remove(self.topic)

    async def send_bytes_async(self, msg: bytes) -> None:
        await self.socket.send(msg)


class Streamer(Publisher):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], Optional[Union[str, bytes, Dict]]],
        fps: int = 45,
    ):
        super().__init__(topic)
        self.running = False
        self.dt: float = 1 / fps
        self.update_func = update_func
        self.manager.submit_task(self.update_loop)
        self.topic_byte = self.topic.encode("utf-8")

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
        try:
            while self.running:
                diff = time.monotonic() - last
                if diff < self.dt:
                    await async_sleep(self.dt - diff)
                last = time.monotonic()
                await self.socket.send(
                    b"".join([self.topic_byte, b":", self.generate_byte_msg()])
                )
        except Exception as e:
            logger.error(f"Error when streaming {self.topic}: {e}")


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
        self.manager.service_list[service_name] = self
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
        pass


class StringService(Service):
    async def send(self, string: str):
        await self.socket.send_string(string)


class BytesService(Service):
    async def send(self, data: bytes):
        await self.socket.send(data)


class DictService(Service):
    async def send(self, data: Dict):
        await self.socket.send_string(dumps(data))
