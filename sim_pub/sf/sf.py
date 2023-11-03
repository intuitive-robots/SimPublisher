from websockets.server import WebSocketServer, WebSocketServerProtocol
import asyncio
import abc

from sim_pub.base import ObjectPublisherBase
from sim_pub.primitive import ObjectStreamer
from .utils import *


class SFObjectPublisher(ObjectPublisherBase):

    def __init__(
        self, 
        id, 
        sim_obj,
        scene,
    ) -> None:
        super().__init__(id)
        self.sim_obj = sim_obj
        self.scene = scene

    @abc.abstractmethod
    def get_obj_param_dict(self) -> dict:
        raise NotImplemented

class SFRigidBodyPublisher(SFObjectPublisher):
    def __init__(self, sim_obj, scene) -> None:
        if hasattr(sim_obj, "name"):
            id = sim_obj.name
        elif hasattr(sim_obj, "object_name"):
            id = getattr(sim_obj, "object_name")
        else:
            raise Exception("Cannot find object name")
        super().__init__(id, sim_obj, scene)
        self.get_obj_pos_fct = self.scene.get_obj_pos
        self.get_obj_quat = self.scene.get_obj_quat

    def get_obj_param_dict(self) -> dict:
        return super().get_obj_param_dict()

    def get_obj_state_dict(self) -> dict:
        return super().get_obj_state_dict()

class SFRobotPublisher(ObjectPublisherBase):
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
            start_stream=self.on_stream,
            close_stream=self.close_stream,
        )

    def create_handler(self, ws: WebSocketServerProtocol):
        return [
            asyncio.create_task(self.receive_handler(ws)),
            asyncio.create_task(self.stream_handler(ws)),
        ]

    def start_stream(self, *args, **kwargs):
        print("Start stream to client")
        self.send_dict(self.getInitParamDict())
        self.on_stream = True

    def close_stream(self, *args, **kwargs):
        print("Close stream to client")
        self.on_stream = False

    async def stream_handler(self, ws: WebSocketServerProtocol):
        while self.connected:
            if not self.on_stream:
                await asyncio.sleep(0.1)
                continue
            new_msg = self.getStateMsg()
            try:
                await self._send_dict_msg(new_msg, ws, 0.02)
            except:
                print("error occured when sending messages!!!!!")
                self.connected = False
                await ws.close()
                break
            finally:
                pass
        print("finish the stream handler")

    def register_callback(self, **kwargs):
        for k, cb in kwargs.items():
            self.register_callback_dict[k] = cb

    async def execute_callback(self, request: dict):
        request_type = request["Type"]
        if request_type in self.register_callback_dict.keys():
            callback = self.register_callback_dict[request_type]
        else:
            print(f"Wrong Request Type for {request_type}!!!")
            return
        callback(request)

    def send_message(self, msg: str):
        self.send_dict(
            {
                "Header": "text_message",
                "TextMessage": msg,
            }
        )
