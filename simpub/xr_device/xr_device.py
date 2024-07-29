import json


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDivece:
    def __init__(self, name):
        self.name = name
        self._state = "off"

    def turn_on(self):
        self._state = "on"

    def turn_off(self):
        self._state = "off"

    def is_on(self):
        return self._state == "on"

    def is_off(self):
        return self._state == "off"

    def __str__(self):
        return f"{self.name} is {self._state}"

    def get_input_data(self) -> InputData:
        raise NotImplementedError

