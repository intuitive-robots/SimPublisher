from websockets.server import WebSocketServer, WebSocketServerProtocol
import asyncio
import abc
from json import loads
from typing import TypedDict, Dict, Union
import mujoco
from mujoco import MjData, MjModel, mj_name2id, mjtObj

from alr_sim.sims.mj_beta import MjScene, MjRobot
from alr_sim.core.sim_object import SimObject
from alr_sim.sims.universal_sim.PrimitiveObjects import PrimitiveObject
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
from alr_sim.core.Robots import RobotBase

from sim_pub.base import SimPubData, SimPubMsg
from sim_pub.primitive import SimStreamer
from sim_pub.utils import *

from geometry import *
from sfmj_simobj_publisher import *

class SFSimPubFactory:

    _publisher_map: Dict[type, type] = {
        PrimitiveObject: SFPrimitiveObjectPublisher,
        YCBMujocoObject: SFYCBObjectPublisher,
        RobotBase: SFPandaPublisher,
    }

    @classmethod
    def create_publisher(
        cls, 
        sim_obj: Union[SimObject, RobotBase], 
        scene: MjScene, 
        **kwargs,
    ) -> SFObjectPublisher:
        pub_type = cls._publisher_map[sim_obj]
        return pub_type(sim_obj, scene, **kwargs)


class SFSimStreamer(SimStreamer):
    
    def __init__(
            self, 
            object_handler_list: list[SFObjectPublisher], 
            scene: Scene,
            host="127.0.0.1", 
            port=8052,
        ) -> None:
        super().__init__(object_handler_list, host, port)

        super().__init__()
        self._object_handler_dict = {
            handler.id: handler for handler in object_handler_list
        }
        self.scene = scene
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
        init_param_msg = SimPubMsg()
        init_param_msg["header"] = "initial_parameter"
        data = SimPubData()
        for item in self.publisher_list:
            item.update_obj_param_dict(data)
        init_param_msg["data"] = data
        await self._send_str_msg_on_loop(dict2encodedstr(init_param_msg))

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
            self.scene.set_obj_pos_and_quat(
                data["pos"], data["rot"], obj_name=id
            )
            self.scene.model.body_gravcomp[body_id] = 1