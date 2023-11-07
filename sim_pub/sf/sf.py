from websockets.server import WebSocketServer, WebSocketServerProtocol
import asyncio
import abc
from json import loads
from mujoco import MjData, MjModel, mj_name2id, mjtObj
from alr_sim.core.sim_object import SimObject
from alr_sim.core.Scene import Scene


import mujoco

from sim_pub.base import ObjectPublisherBase
from sim_pub.primitive import ObjectStreamer
from sim_pub.utils import *
from sim_pub.geometry import *


class SFObjectPublisher(ObjectPublisherBase):

    def __init__(
        self,
        id: str,
        sim_obj: SimObject,
        scene: Scene,
    ) -> None:
        super().__init__(id)
        self.sim_obj = sim_obj
        self.scene = scene

class SFRigidBodyPublisher(SFObjectPublisher):
    def __init__(
        self, 
        sim_obj: SimObject, 
        scene: Scene,
        size: list[float] = [1, 1, 1],
        rgba: list[float] = [-1, -1, -1, 1],
        static: bool = False,
        interactable: bool = False,
    ) -> None:
        if hasattr(sim_obj, "name"):
            id = sim_obj.name
        elif hasattr(sim_obj, "object_name"):
            id = getattr(sim_obj, "object_name")
        else:
            raise Exception("Cannot find object name")
        super().__init__(id, sim_obj, scene)
        self.get_obj_pos_fct = self.scene.get_obj_pos
        self.get_obj_quat_fct = self.scene.get_obj_quat

    def get_obj_param_dict(self) -> dict:
        return {
            "Header": "initial_parameter",
            "Data": {
                "pos": list(mj2unity_pos(self.get_obj_pos_fct(self.sim_obj))),
                "rot": list(mj2unity_quat(self.get_obj_quat_fct(self.sim_obj))),
                "size": mj2unity_size(self.sim_obj),
                "rgba": [-1, -1, -1, 1] if not hasattr(self.sim_obj, "rgba") else self.sim_obj.rgba,
                "rot_offset": [0, 0, 0]
                if not hasattr(self.sim_obj, "rot_offset")
                else getattr(self.sim_obl, "rot_offset"),
            },
        }

    def get_obj_state_dict(self) -> dict:
        return {
            "Header": "sim_state",
            "data": {
                "pos": list(mj2unity_pos(self.get_obj_pos_fct(self.obj))),
                "rot": list(mj2unity_quat(self.get_obj_quat_fct(self.obj))),
            }
        }

class SFPandaPublisher(SFObjectPublisher):
    pass


class SFObjectStreamer(ObjectStreamer):
    
    def __init__(
            self, 
            object_handler_list: list[SFObjectPublisher], 
            host="127.0.0.1", 
            port=8052,
        ) -> None:
        super().__init__(object_handler_list, host, port)

        super().__init__()
        self._object_handler_dict = {handler.id: handler for handler in object_handler_list}
        # flags
        self.on_stream = False

        # defaule register function
        self.register_callback_dict = dict()
        self.register_callback(
            start_stream=self.start_stream,
            close_stream=self.close_stream,
            manipulate_objects=self.update_manipulated_object,
        )

    def create_handler(self, ws: WebSocketServerProtocol):
        return [
            asyncio.create_task(self.receive_handler(ws)),
            asyncio.create_task(self.stream_handler(ws)),
        ]

    async def on_start_stream(self):
        await super().on_start_stream()
        init_param_dict = {
            "Header": "initial_parameter",
            "Data": [item.get_obj_param_dict() for item in self.publisher_list],
        }
        await self._send_str_msg_on_loop(dict2encodedstr(init_param_dict))

    def register_callback(self, **kwargs):
        for k, cb in kwargs.items():
            self.register_callback_dict[k] = cb

    async def process_message(self, msg: str) -> None:
        await self.execute_callback(loads(msg))

    async def execute_callback(self, request: dict):
        request_type = request["message_type"]
        if request_type in self.register_callback_dict.keys():
            callback = self.register_callback_dict[request_type]
        else:
            print(f"Wrong Request Type for {request_type}!!!")
            return
        callback(request)

    def update_manipulated_object(self, msg):
        for id, obj_data in msg["manipulationData"].items():
            body_id = mj_name2id(self.scene.model, mjtObj.mjOBJ_BODY, id)
            if obj_data is None:
                self.scene.model.body_gravcomp[body_id] = 0
                continue
            data = obj_data["data"]
            self.scene.set_obj_pos_and_quat(data["pos"], data["rot"], obj_name=id)
            self.scene.model.body_gravcomp[body_id] = 1