from typing import TypedDict
import json

from ..core.publisher import Publisher
from .xr_device import XRDevice


class MetaQuest3Hand(TypedDict):
    pos: list[float]
    rot: list[float]
    index_trigger: bool
    hand_trigger: bool


class MetaQuest3InputData(TypedDict):
    left: MetaQuest3Hand
    right: MetaQuest3Hand
    A: bool
    B: bool
    X: bool
    Y: bool

# TODO: Vibration Data Structure
class Vibration(TypedDict):
    hand: str
    amplitude: float


class MetaQuest3(XRDevice):
    def __init__(
        self,
        device_name: str,
    ) -> None:
        super().__init__(device_name)
        self.input_data: MetaQuest3InputData = None
        self.input_subscriber = self.register_topic_callback(
            f"{device_name}/InputData", self.update
        )
        self.viborate_publisher = Publisher(f"{device_name}/Vibration")

    def update(self, data: str):
        self.input_data = json.loads(data)

    def get_input_data(self) -> MetaQuest3InputData:
        return self.input_data

    # TODO: Vibration Data Structure
    def publish_vibrate(self, hand: str = "right"):
        self.viborate_publisher.publish_string(hand)
