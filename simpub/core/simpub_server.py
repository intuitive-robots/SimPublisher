from __future__ import annotations
import abc
from typing import Dict, List, Optional
import json
import zmq

from ..simdata import SimScene
from .net_manager import init_net_manager
from .net_manager import Streamer, BytesService, HostInfo
from .log import logger
from .utils import send_message


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

    def shutdown(self):
        self.net_manager.shutdown()


class MsgServer(ServerBase):

    def initialize(self):
        pass


class SimPublisher(ServerBase):

    def __init__(
        self,
        sim_scene: SimScene,
        host: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
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
        self.net_manager.register_service.on_trigger_events.append(
            self.on_xr_client_registered
        )

    def initialize(self):
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        # self.scene_service = BytesService("Scene", self._on_scene_request)
        self.asset_service = BytesService("Asset", self._on_asset_request)

    def on_xr_client_registered(self, msg: str):
        xr_info: HostInfo = json.loads(msg)
        if "LoadSimScene" in xr_info["serviceList"]:
            scene_string = f"LoadSimScene:{self.sim_scene.to_string()}"
            req_socket = self.net_manager.create_socket(zmq.REQ)
            req_socket.connect(
                f"tcp://{xr_info['ip']}:{xr_info['servicePort']}"
            )
            self.net_manager.submit_task(
                send_message, scene_string, req_socket
            )
            logger.info(f"Send scene to {xr_info['name']}")
            req_socket.close()

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
