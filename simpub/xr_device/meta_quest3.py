import json
from typing import TypedDict

from simpub.server import SimPublisher
from .xr_device import XRDevice


class MetaQuest3InputData(TypedDict):
    left__pos: list[float]
    left_rot: list[float]
    left_index_trigger: float
    left_hand_trigger: float
    right_pos: list[float]
    right_rot: list[float]
    right_index_trigger: float
    right_hand_trigger: float
    A: bool
    B: bool
    X: bool
    Y: bool


class MetaQuest3(XRDevice):
    def __init__(
        self,
        publisher: SimPublisher,
        device: str,
        port: str = "7723",
    ) -> None:
        super().__init__(publisher, addr, port)
        self.input_data: MetaQuest3InputData = None

    def setup_subscribe_topic(self):
        self.subscribe_topic("MetaQuest3/InputData", self.update)

    def update(self, data: MetaQuest3InputData):
        self.input_data = data

    def get_input_data(self) -> MetaQuest3InputData:
        return self.input_data
