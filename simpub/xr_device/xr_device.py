import json
from typing import Callable

from simpub.server import SimPublisher
from simpub.server import SubscribeTask


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDivece:
    def __init__(
        self,
        publisher: SimPublisher,
        addr: str = "127.0.0.1",
        port: str = "7723",
    ) -> None:
        self._state = "off"
        self.publisher = publisher
        self.addr = addr
        self.port = port
        self.sub_task = SubscribeTask(
            self.publisher.zmqContext,
            addr=self.addr,
            port=self.port,
        )
        self.publisher.add_task(self.sub_task)
        self.subscribe_topic("UnityLog", self.print_log)
        self.setup_subscribe_topic()

    def print_log(self, log: str):
        print(f"UnityLog: {log}")

    def subscribe_topic(
        self,
        topic: str,
        callback: Callable[[str], None],
    ) -> None:
        self.sub_task.register_callback(topic, callback)

    def setup_subscribe_topic(self):
        raise NotImplementedError

    def is_on(self):
        return self._state == "on"

    def is_off(self):
        return self._state == "off"

    def get_input_data(self) -> InputData:
        raise NotImplementedError
