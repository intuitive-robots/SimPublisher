from __future__ import annotations
import abc
from typing import Dict, List, Optional, Set
from asyncio import sleep as asyncio_sleep
import traceback

from ..simdata import SimScene
from .net_manager import NodeManager, Streamer, StrBytesService, init_node
from .log import logger
from .utils import send_request, HashIdentifier


class ServerBase(abc.ABC):

    def __init__(self, ip_addr: str, node_name: str):
        self.ip_addr: str = ip_addr
        self.net_manager = init_node(ip_addr, node_name)
        self.initialize()
        self.net_manager.start_node_broadcast()

    def spin(self):
        self.net_manager.spin()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    def shutdown(self):
        self.net_manager.stop_node()


class MsgServer(ServerBase):

    def initialize(self):
        pass


class SimPublisher(ServerBase):

    def __init__(
        self,
        sim_scene: SimScene,
        ip_addr: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
        fps: int = 30,
    ) -> None:
        self.sim_scene = sim_scene
        self.fps = fps
        if no_rendered_objects is None:
            self.no_rendered_objects = []
        else:
            self.no_rendered_objects = no_rendered_objects
        if no_tracked_objects is None:
            self.no_tracked_objects = []
        else:
            self.no_tracked_objects = no_tracked_objects
        super().__init__(ip_addr, "SimPublisher")

    def initialize(self) -> None:
        self.scene_update_streamer = Streamer(
            topic_name="SceneUpdate",
            update_func=self.get_update,
            fps=self.fps,
            start_streaming=True)
        self.asset_service = StrBytesService("Asset", self._on_asset_request)
        self.xr_device_set: Set[HashIdentifier] = set()
        self.net_manager.submit_task(self.search_xr_device, self.net_manager)

    async def search_xr_device(self, node: NodeManager):
        while node.running:
            for xr_info in node.nodes_info_manager.nodes_info.values():
                if xr_info["nodeID"] in self.xr_device_set:
                    continue
                if "LoadSimScene" not in xr_info["serviceList"]:
                    continue
                try:
                    self.xr_device_set.add(xr_info["nodeID"])
                    scene_string = f"LoadSimScene|{self.sim_scene.to_string()}"
                    ip, port = xr_info["addr"]["ip"], xr_info["servicePort"]
                    self.net_manager.submit_task(
                        send_request,
                        scene_string,
                        f"tcp://{ip}:{port}",
                        self.net_manager.zmq_context
                    )
                    logger.info(f"The Scene is sent to {xr_info['name']}")
                except Exception as e:
                    logger.error(f"Error when sending scene to xr device: {e}")
                    traceback.print_exc()
            await asyncio_sleep(0.5)

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
