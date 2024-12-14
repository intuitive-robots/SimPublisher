from __future__ import annotations
import abc
from typing import Dict, List, Optional, Set
from asyncio import sleep as asyncio_sleep

from ..simdata import SimScene
from .net_manager import NodeManager, NodeInfo, init_node
from .net_component import Streamer, BytesService
from .log import logger
from .utils import send_request


class ServerBase(abc.ABC):

    def __init__(self, host: str = "127.0.0.1"):
        self.host: str = host
        self.net_manager = init_node(host)
        # self.initialize()
    #     self.net_manager.start()

    # def join(self):
    #     self.net_manager.join()

    # @abc.abstractmethod
    # def initialize(self):
    #     raise NotImplementedError

    # def shutdown(self):
    #     self.net_manager.shutdown()


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
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        self.asset_service = BytesService("Asset", self._on_asset_request)
        self.xr_device_set: Set[NodeInfo] = set()
        self.net_manager.submit_task(self.search_xr_device, self.net_manager)

    async def search_xr_device(self, node: NodeManager):
        print("Start searching xr device")
        while node.running:
            xr_info = node.nodes_info_manager.check_service("LoadSimScene")
            print(f"xr_info: {xr_info}")
            if xr_info is not None and xr_info not in self.xr_device_set:
                self.xr_device_set.add(xr_info)
                scene_string = f"LoadSimScene|{self.sim_scene.to_string()}"
                self.net_manager.submit_task(
                    send_request,
                    scene_string,
                    f"tcp://{xr_info['addr']['ip']}:{xr_info['servicePort']}",
                    self.net_manager.zmq_context
                )
                print(f"Send scene to {xr_info['name']}")
                logger.info(f"Send scene to {xr_info['name']}")
            await asyncio_sleep(0.5)

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
