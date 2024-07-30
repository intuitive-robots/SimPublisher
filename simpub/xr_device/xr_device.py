import json

from simpub.server import SimPublisher


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDivece:
    def __init__(
        self,
        publisher: SimPublisher,
        addr: str = "127.0.0.1",
        port: str = "7725",
    ) -> None:
        self._state = "off"
        self.publisher = publisher
        self.addr = addr
        self.port = port
        self.create_task()

    def create_task(self):
        raise NotImplementedError

    def is_on(self):
        return self._state == "on"

    def is_off(self):
        return self._state == "off"

    def get_input_data(self) -> InputData:
        raise NotImplementedError
