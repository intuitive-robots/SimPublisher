from typing import Optional, TypedDict, Callable, Dict, List
import json
import enum

# from asyncio import sleep as async_sleep

from simpub.core.net_component import Subscriber
from .xr_device import XRDevice


class MetaQuest3MotionController(TypedDict):
    pos: List[float]
    rot: List[float]
    index_trigger: bool
    hand_trigger: bool


class MetaQuest3MotionControllerData(TypedDict):
    left: MetaQuest3MotionController
    right: MetaQuest3MotionController
    A: bool
    B: bool
    X: bool
    Y: bool


# TODO: Vibration Data Structure
# class Vibration(TypedDict):
#     hand: str
#     amplitude: float


class MetaQuest3Bone(TypedDict):
    pos: List[float]
    rot: List[float]


class MetaQuest3Hand(TypedDict):
    bones: List[MetaQuest3Bone]
    # TODO: hand gesture data can be added here


class MetaQuest3HandTrackingData(TypedDict):
    leftHand: MetaQuest3Hand
    rightHand: MetaQuest3Hand


class MetaQuest3HandBoneID(enum.IntEnum):
    XRHand_Palm = 0
    XRHand_Wrist = 1
    XRHand_ThumbMetacarpal = 2
    XRHand_ThumbProximal = 3
    XRHand_ThumbDistal = 4
    XRHand_ThumbTip = 5
    XRHand_IndexMetacarpal = 6
    XRHand_IndexProximal = 7
    XRHand_IndexIntermediate = 8
    XRHand_IndexDistal = 9
    XRHand_IndexTip = 10
    XRHand_MiddleMetacarpal = 11
    XRHand_MiddleProximal = 12
    XRHand_MiddleIntermediate = 13
    XRHand_MiddleDistal = 14
    XRHand_MiddleTip = 15
    XRHand_RingMetacarpal = 16
    XRHand_RingProximal = 17
    XRHand_RingIntermediate = 18
    XRHand_RingDistal = 19
    XRHand_RingTip = 20
    XRHand_LittleMetacarpal = 21
    XRHand_LittleProximal = 22
    XRHand_LittleIntermediate = 23
    XRHand_LittleDistal = 24
    XRHand_LittleTip = 25


class MetaQuest3(XRDevice):
    def __init__(
        self,
        device_name: str,
    ) -> None:
        super().__init__(device_name)
        self.last_input_data: Optional[MetaQuest3MotionControllerData] = None
        self.motion_controller_data: Optional[
            MetaQuest3MotionControllerData
        ] = None
        self.hand_tracking_data: Optional[MetaQuest3HandTrackingData] = None
        self.sub_list.append(
            Subscriber("MotionController", self.update_motion_controller)
        )
        self.sub_list.append(
            Subscriber("HandTracking", self.update_hand_tracking)
        )
        # self.start_vib_pub = Publisher(f"{device_name}/StartVibration")
        # self.stop_vib_pub = Publisher(f"{device_name}/StopVibration")
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

    def update_motion_controller(self, data: str):
        self.last_input_data = self.input_data
        self.input_data = json.loads(data)
        if self.last_input_data is None:
            return
        if self.input_data is None:
            return
        for button, callbacks in self.button_press_event.items():
            if self.input_data[button] and not self.last_input_data[button]:
                [callback() for callback in callbacks]
        left_hand = self.input_data["left"]
        last_left_hand = self.last_input_data["left"]
        for trigger, callbacks in self.left_trigger_press_event.items():
            if left_hand[trigger] and not last_left_hand[trigger]:
                [callback() for callback in callbacks]
        for trigger, callbacks in self.left_trigger_release_event.items():
            if not left_hand[trigger] and last_left_hand[trigger]:
                [callback() for callback in callbacks]
        right_hand = self.input_data["right"]
        last_right_hand = self.last_input_data["right"]
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

    def get_controller_data(self) -> Optional[MetaQuest3MotionControllerData]:
        return self.motion_controller_data

    def update_hand_tracking(self, data: str):
        self.hand_tracking_data: MetaQuest3HandTrackingData = json.loads(data)

    def get_hand_tracking_data(self) -> Optional[MetaQuest3HandTrackingData]:
        return self.hand_tracking_data

    # # TODO: Vibration Data Structure
    # def start_vibration(self, hand: str = "right", duration=0.5):
    #     if not self.on_vibration[hand]:
    #         self.on_vibration[hand] = True
    #         self.manager.submit_asyncio_task(
    #             self.start_vibration_async,
    #             hand,
    #             duration,
    #         )

    # async def start_vibration_async(self, hand: str = "right", duration=0.5):
    #     self.start_vib_pub.publish_string(hand)
    #     if duration > 1.5:
    #         while duration < 0:
    #             await async_sleep(1.5)
    #             self.start_vib_pub.publish_string(hand)
    #             duration -= 1.5
    #     else:
    #         await async_sleep(duration)
    #     self.stop_vibration(hand)

    # def stop_vibration(self, hand: str = "right"):
    #     if self.on_vibration[hand]:
    #         self.on_vibration[hand] = False
    #     self.stop_vib_pub.publish_string(hand)
