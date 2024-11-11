from __future__ import annotations
import abc
from typing import Dict, List

from simpub.simdata import SimScene
from .net_manager import init_net_manager, asycnc_sleep
from .net_manager import Streamer, BytesService, SimPubClient
# from .net_manager import ServerPort
# from .log import logger


class ServerBase(abc.ABC):

    def __init__(self, host: str = "127.0.0.1"):
        self.host: str = host
        self.net_manager = init_net_manager(host)
        self.initialize()
        self.net_manager.start()

    def join(self):
        self.net_manager.join()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError


class MsgServer(ServerBase):

    def initialize(self):
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
        self.net_manager.on_client_registered.append(
            self.on_xr_client_registered
        )
        super().__init__(host)

    def initialize(self):
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        self.scene_service = BytesService("Scene", self._on_scene_request)
        self.asset_service = BytesService("Asset", self._on_asset_request)

    def on_xr_client_registered(self, client: SimPubClient):
        if "CreateScene" in client.info["services"]:
            client.req_socket.send_string("CreateScene:")

    def _on_scene_request(self, req: str) -> str:
        return self.sim_scene.to_string()

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    async def check_and_send_scene_update(self):
        for obj in self.net_manager.clients_info.values():
            if obj.needs_update:
                await self.scene_update_streamer.send()
                break
            await asycnc_sleep(0.05)

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
