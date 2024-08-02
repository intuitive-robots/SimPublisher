from __future__ import annotations
import abc
from typing import Dict, List
import zmq

from simpub.simdata import SimScene
from .net_manager import init_net_manager
from .publisher import Streamer
from .service import Service


class ServerBase(abc.ABC):

    def __init__(self, host: str = "127.0.0.1"):
        self.host: str = host
        self.net_manager = init_net_manager(host)


class MsgServer(ServerBase):
    pass


class SimPublisher(ServerBase):

    def __init__(
        self,
        sim_scene: SimScene,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        self.sim_scene = sim_scene
        if no_rendered_objects is None:
            self.no_rendered_objects = []
        else:
            self.no_rendered_objects = no_rendered_objects
        if no_tracked_objects is None:
            self.no_tracked_objects = []
        else:
            self.no_tracked_objects = no_tracked_objects
        super().__init__(host)
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        self.scene_service = Service("Scene", self._on_scene_request)
        self.asset_service = Service("Asset", self._on_asset_request)

    def _on_scene_request(self, socket: zmq.Socket, tag: str):
        socket.send_string(self.sim_scene.to_string())

    def _on_asset_request(self, socket: zmq.Socket, tag: str):
        if tag not in self.sim_scene.raw_data:
            print("Received invalid data request")
            return
        socket.send(self.sim_scene.raw_data[tag])

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
