from dataclasses import dataclass
import json

from .xr_device import XRDivece


@dataclass
class MetaQuest3InputData:
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


class MetaQuest3(XRDivece):
    def __init__(self, name, xray, quest):
        super().__init__(name, xray)
        self.quest = quest
        self.input_data_str: str = ""

    def get_input_data(self) -> MetaQuest3InputData:
        json_dict = json.loads(self.input_data_str)
        return MetaQuest3InputData(
            left__pos=json_dict["left__pos"],
            left_rot=json_dict["left_rot"],
            left_index_trigger=json_dict["left_index_trigger"],
            left_hand_trigger=json_dict["left_hand_trigger"],
            right_pos=json_dict["right_pos"],
            right_rot=json_dict["right_rot"],
            right_index_trigger=json_dict["right_index_trigger"],
            right_hand_trigger=json_dict["right_hand_trigger"],
            A=json_dict["A"],
            B=json_dict["B"],
            X=json_dict["X"],
            Y=json_dict["Y"],
        )
