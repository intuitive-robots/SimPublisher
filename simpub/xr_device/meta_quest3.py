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
    def __init__(self):
        super().__init__()
        self.input_data: dict = None

    def update(self, data: str):
        # print(data)
        if data == "MetaQuest3/InputData":
            return
        self.input_data = json.loads(data)
        # print("here")

    def get_input_data(self) -> MetaQuest3InputData:
        if self.input_data is None:
            return None
        input_data = self.input_data
        return MetaQuest3InputData(
            left__pos=input_data["left_pos"],
            left_rot=input_data["left_rot"],
            left_index_trigger=input_data["left_index_trigger"],
            left_hand_trigger=input_data["left_hand_trigger"],
            right_pos=input_data["right_pos"],
            right_rot=input_data["right_rot"],
            right_index_trigger=input_data["right_index_trigger"],
            right_hand_trigger=input_data["right_hand_trigger"],
            A=input_data["A"],
            B=input_data["B"],
            X=input_data["X"],
            Y=input_data["Y"],
        )
