from typing import TypedDict, Callable, Dict, List
import json

from simpub.core.net_manager import Publisher
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
        self.last_input_data: MetaQuest3InputData = None
        self.input_data: MetaQuest3InputData = None
        self.input_subscriber = self.register_topic_callback(
            f"{device_name}/InputData", self.update
        )
        self.viborate_publisher = Publisher(f"{device_name}/Vibration")
        self.button_trigger_event: Dict[str, List[Callable]] = {
            "A": [],
            "B": [],
            "X": [],
            "Y": [],
        }
        self.on_vibration = {"left": False, "right": False}

    def update(self, data: str):
        self.last_input_data = self.input_data
        self.input_data = json.loads(data)
        if self.last_input_data is None:
            return
        for button, callbacks in self.button_trigger_event.items():
            if self.input_data[button] and not self.last_input_data[button]:
                [callback() for callback in callbacks]

    def register_button_trigger_event(self, button: str, callback: Callable):
        # button should be one of A, B, X, Y
        self.button_trigger_event[button].append(callback)

    def get_input_data(self) -> MetaQuest3InputData:
        return self.input_data

    # TODO: Vibration Data Structure
    def start_vibrate(self, hand: str = "right"):
        if not self.on_vibration[hand]:
            self.viborate_publisher.publish_string(hand)
            self.on_vibration[hand] = True

    def stop_vibrate(self, hand: str = "right"):
        if self.on_vibration[hand]:
            self.on_vibration[hand] = False
