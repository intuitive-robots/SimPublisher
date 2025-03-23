from typing import Optional, TypedDict, Callable, Dict, List
import json
from asyncio import sleep as async_sleep

from simpub.core.net_component import Publisher
from .xr_device import XRDevice


class MetaQuest3Hand(TypedDict):
    pos: List[float]
    rot: List[float]
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
        self.last_input_data: Optional[MetaQuest3InputData] = None
        self.input_data: Optional[MetaQuest3InputData] = None
        self.input_subscriber = self.register_topic_callback(
            f"{device_name}/InputData", self.update
        )
        self.start_vib_pub = Publisher(f"{device_name}/StartVibration")
        self.stop_vib_pub = Publisher(f"{device_name}/StopVibration")
        self.button_press_event: Dict[str, List[Callable]] = {
            "A": [],
            "B": [],
            "X": [],
            "Y": [],
        }
        self.left_trigger_press_event: Dict[str, List[Callable]] = {
            "hand_trigger": [],
            "index_trigger": [],
        }
        self.left_trigger_release_event: Dict[str, List[Callable]] = {
            "hand_trigger": [],
            "index_trigger": [],
        }
        self.right_trigger_press_event: Dict[str, List[Callable]] = {
            "hand_trigger": [],
            "index_trigger": [],
        }
        self.right_trigger_release_event: Dict[str, List[Callable]] = {
            "hand_trigger": [],
            "index_trigger": [],
        }
        self.on_vibration = {"left": False, "right": False}

    def update(self, data: str):
        self.last_input_data = self.input_data
        self.input_data = json.loads(data)
        if self.last_input_data is None:
            return
        if self.input_data is None:
            return
        for button, callbacks in self.button_press_event.items():
            if self.input_data[button] and not self.last_input_data[button]:
                [callback() for callback in callbacks]
        # hand controller info with order in unity coordinates (left handed):
        # pos: z, -x, y & rot: -z, x, -y, w
        left_hand = self.input_data["left"]
        last_left_hand = self.last_input_data["left"]
        # to isaac sim pos
        # x := -z, y:= x, z:= y
        left_hand["pos"] = [
            -left_hand["pos"][0],
            -left_hand["pos"][1],
            left_hand["pos"][2],
        ]
        # to isaac sim rot:
        # w:= w, x:= z, y:= -x, z:= -y
        left_hand["rot"] = [
            left_hand["rot"][3],
            -left_hand["rot"][0],
            -left_hand["rot"][1],
            left_hand["rot"][2],
        ]        
        for trigger, callbacks in self.left_trigger_press_event.items():
            if left_hand[trigger] and not last_left_hand[trigger]:
                [callback() for callback in callbacks]
        for trigger, callbacks in self.left_trigger_release_event.items():
            if not left_hand[trigger] and last_left_hand[trigger]:
                [callback() for callback in callbacks]
        # hand controller info with order in unity coordinates (left handed):
        # pos: z, -x, y & rot: -z, x, -y, w        
        right_hand = self.input_data["right"]
        last_right_hand = self.last_input_data["right"]
        # to isaac sim pos:
        # x := -z, y:= x, z:= y
        right_hand["pos"] = [
            -right_hand["pos"][0],
            -right_hand["pos"][1],
            right_hand["pos"][2],
        ]
        # to isaac sim rot:
        # w:= w, x:= z, y:= -x, z:= -y
        right_hand["rot"] = [
            right_hand["rot"][3],
            -right_hand["rot"][0],
            -right_hand["rot"][1],
            right_hand["rot"][2],
        ]        
        for trigger, callbacks in self.right_trigger_press_event.items():
            if right_hand[trigger] and not last_right_hand[trigger]:
                [callback() for callback in callbacks]
        for trigger, callbacks in self.right_trigger_release_event.items():
            if not right_hand[trigger] and last_right_hand[trigger]:
                [callback() for callback in callbacks]

    def register_button_press_event(self, button: str, callback: Callable):
        # button should be one of A, B, X, Y
        self.button_press_event[button].append(callback)

    def register_trigger_press_event(
        self, trigger: str, hand: str, callback: Callable
    ):
        # hand should be one of left or right
        # trigger should be one of hand_trigger or index_trigger
        if hand == "left":
            self.left_trigger_press_event[trigger].append(callback)
        elif hand == "right":
            self.right_trigger_press_event[trigger].append(callback)
        else:
            raise ValueError("Invalid hand")

    def register_trigger_release_event(
        self, trigger: str, hand: str, callback: Callable
    ):
        # hand should be one of
        # left_hand, left_trigger, right_hand, right_trigger
        if hand == "left":
            self.left_trigger_release_event[trigger].append(callback)
        elif hand == "right":
            self.right_trigger_release_event[trigger].append(callback)
        else:
            raise ValueError("Invalid hand")

    def get_input_data(self) -> Optional[MetaQuest3InputData]:
        return self.input_data

    # TODO: Vibration Data Structure
    def start_vibration(self, hand: str = "right", duration=0.5):
        if not self.on_vibration[hand]:
            self.on_vibration[hand] = True
            self.manager.submit_task(
                self.start_vibration_async, hand, duration,
            )

    async def start_vibration_async(self, hand: str = "right", duration=0.5):
        self.start_vib_pub.publish_string(hand)
        if duration > 1.5:
            while duration < 0:
                await async_sleep(1.5)
                self.start_vib_pub.publish_string(hand)
                duration -= 1.5
        else:
            await async_sleep(duration)
        self.stop_vibration(hand)

    def stop_vibration(self, hand: str = "right"):
        if self.on_vibration[hand]:
            self.on_vibration[hand] = False
        self.stop_vib_pub.publish_string(hand)
