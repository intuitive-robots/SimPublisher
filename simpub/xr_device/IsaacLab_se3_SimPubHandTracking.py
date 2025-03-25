from collections.abc import Callable

import numpy as np
from isaaclab.devices import DeviceBase
from isaacsim.core.utils.rotations import quat_to_euler_angles
from simpub.xr_device.meta_quest3 import MetaQuest3


class Se3SimPubHandTracking(DeviceBase):
    def __init__(self, device_name="ALRMetaQuest3", hand="right"):
        self.meta_quest3 = MetaQuest3(device_name)
        assert hand == "right" or hand == "left", (
            "hand={} invalid. Only right and left are supported".format(hand)
        )
        self.hand = hand
        self._close_gripper = False
        self._tracking_active = False
        self._lock_rot = False

        self.add_callback("B", self._toggle_gripper_state)
        self.add_callback("right:index_trigger@press", self._toggle_tracking)
        self.add_callback("right:index_trigger@release", self._toggle_tracking)
        self.add_callback("left:index_trigger@press", self._toggle_lock_rot)
        self.add_callback("left:index_trigger@release", self._toggle_lock_rot)

    def reset(self):
        """Reset the internals."""
        self._close_gripper = False
        return

    def add_callback(
        self,
        key: str,
        func: Callable,
    ):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        # get hand
        if ":" in key:
            hand, action = key.split(":")
        else:
            hand = "right"
            action = key
        # get button press type
        if "@" in action:
            key, press_type = action.split("@")
        else:
            key = action
            press_type = "press"
        # register callback
        if key in ["A", "B", "X", "Y"]:
            self.meta_quest3.register_button_press_event(key, func)
        elif key in ["hand_trigger", "index_trigger"]:
            if press_type == "press":
                self.meta_quest3.register_trigger_press_event(key, hand, func)
            else:
                self.meta_quest3.register_trigger_release_event(key, hand, func)
        return

    def _get_input_data(self):
        input_data = self.meta_quest3.get_input_data()

        if input_data is not None:
            # hand controller info with order in unity coords:
            # pos: z, -x, y /  rot: -z, x, -y, w
            # to isaac sim coords
            # x := -z, y:= x, z:= y
            input_data["left"]["pos"] = [
                -input_data["left"]["pos"][0],
                -input_data["left"]["pos"][1],
                input_data["left"]["pos"][2],
            ]
            # to isaac sim rot:
            # w:= w, x:= z, y:= -x, z:= -y
            input_data["left"]["rot"] = [
                input_data["left"]["rot"][3],
                -input_data["left"]["rot"][0],
                -input_data["left"]["rot"][1],
                input_data["left"]["rot"][2],
            ]

            # same for right hand
            input_data["right"]["pos"] = [
                -input_data["right"]["pos"][0],
                -input_data["right"]["pos"][1],
                input_data["right"]["pos"][2],
            ]
            input_data["right"]["rot"] = [
                input_data["right"]["rot"][3],
                -input_data["right"]["rot"][0],
                -input_data["right"]["rot"][1],
                input_data["right"]["rot"][2],
            ]

        return input_data

    def _toggle_gripper_state(self):
        self._close_gripper = not self._close_gripper

    def _toggle_tracking(self):
        self._tracking_active = not self._tracking_active

    def _toggle_lock_rot(self):
        self._lock_rot = not self._lock_rot


class Se3SimPubHandTrackingRel(Se3SimPubHandTracking):
    def __init__(
        self,
        device_name="ALRMetaQuest3",
        hand="right",
        # factor of 10 corresponds to 1:1 tracking
        delta_pos_scale_factor=10,
        delta_rot_scale_factor=10,
    ):
        super().__init__(device_name, hand)
        self._pos = np.zeros(3)
        self._rot = np.zeros(3)
        self._delta_pos_scale_factor = delta_pos_scale_factor
        self._delta_rot_scale_factor = delta_rot_scale_factor

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the joystick event state.
        Returns:
            The processed output form the joystick.
        """
        input_data = self._get_input_data()

        d_rot = np.zeros(3)
        d_pos = np.zeros(3)

        if input_data is not None:
            cur_rot = quat_to_euler_angles(np.asarray(input_data[self.hand]["rot"]))
            if self._tracking_active:
                # calculate position difference
                d_pos = np.asarray(input_data[self.hand]["pos"]) - self._pos
                d_pos *= self._delta_pos_scale_factor
                # calculate rotation difference
                if not self._lock_rot:
                    d_rot = cur_rot - self._rot
                    d_rot *= self._delta_rot_scale_factor

            # store values needed to calculate offset next update cycle
            self._pos = input_data[self.hand]["pos"]
            self._rot = cur_rot

        return np.concatenate([d_pos, d_rot]), self._close_gripper
