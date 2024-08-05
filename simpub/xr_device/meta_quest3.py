from typing import TypedDict

from ..core.subscriber import Subscriber
from ..core.publisher import Publisher
from .xr_device import XRDevice


class MetaQuest3Hand(TypedDict):
    pos: list[float]
    rot: list[float]
    index_trigger: float
    hand_trigger: float


class MetaQuest3InputData(TypedDict):
    left: MetaQuest3Hand
    right: MetaQuest3Hand
    A: bool
    B: bool
    X: bool
    Y: bool


class Viborate(TypedDict):
    hand: str
    intensity: float


class MetaQuest3(XRDevice):
    def __init__(
        self,
        device_name: str,
    ) -> None:
        super().__init__(device_name)
        self.input_data: MetaQuest3InputData = None
        self.input_subscriber = Subscriber(
            f"{device_name}/InputData", self.update
        )
        self.viborate_publisher = Publisher(f"{device_name}/Vibration")

    def update(self, data: MetaQuest3InputData):
        self.input_data = data

    def get_input_data(self) -> MetaQuest3InputData:
        return self.input_data

    def publish_viborate(self, signal: Viborate):
        self.viborate_publisher.publish(signal)
