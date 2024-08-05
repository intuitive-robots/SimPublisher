from typing import Dict, Callable
import time
from json import dumps

from .net_manager import ConnectionAbstract
from .log import logger


class Publisher(ConnectionAbstract):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.socket = self.manager.pub_socket
        self.manager.register_local_topic(self.topic)
        logger.info(f"Publisher for topic {self.topic} is ready")

    def publish(self, data: Dict):
        self.socket.send_json(data)

    def on_shutdown(self):
        super().on_shutdown()


class Streamer(Publisher):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], Dict],
        fps: int = 45,
    ):
        super().__init__(topic)
        self.running = False
        self.dt: float = 1 / fps
        self.update_func = update_func
        self.manager.submit_task(self.update_loop)

    def update_loop(self):
        self.running = True
        last = 0.0
        while self.running:
            diff = time.monotonic() - last
            if diff < self.dt:
                time.sleep(self.dt - diff)
            last = time.monotonic()
            msg = {
                "updateData": self.update_func(),
                "time": time.monotonic()
            }
            self.socket.send_string(f"{self.topic}:{dumps(msg)}")