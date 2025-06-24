from __future__ import annotations
import abc
from typing import Dict, List, Optional, Set
from asyncio import sleep as asyncio_sleep
import traceback

from ..simdata import SimScene, SimObject, SimVisual
from .net_manager import XRNodeManager, init_xr_node_manager
from .log import logger
from .utils import send_string_request, HashIdentifier, XRNodeInfo


class ServerBase(abc.ABC):

    def __init__(self, ip_addr: str):
        self.ip_addr: str = ip_addr
        self.node_manager = init_xr_node_manager(ip_addr)
        self.initialize()
        self.node_manager.start_discover_node_loop()

    def spin(self):
        self.node_manager.spin()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    def shutdown(self):
        self.node_manager.stop_node()


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
        super().__init__(ip_addr)

    def initialize(self) -> None:
        # self.scene_update_streamer = Streamer(
        #     topic_name="SceneUpdate",
        #     update_func=self.get_update,
        #     fps=self.fps,
        #     start_streaming=True)
        # self.asset_service = StrBytesService("Asset", self._on_asset_request)
        self.xr_device_set: Set[HashIdentifier] = set()
        self.node_manager.submit_asyncio_task(
            self.search_xr_device, self.node_manager
        )

    async def search_xr_device(self, node_manager: XRNodeManager):
        while node_manager.running:
            for xr_node_id in node_manager.xr_nodes_info:
                if xr_node_id in self.xr_device_set:
                    continue
                try:
                    self.xr_device_set.add(xr_node_id)
                    self.send_scene_to_xr_device(
                        node_manager.xr_nodes_info[xr_node_id]
                    )
                    # scene_string = f"LoadSimScene|{self.sim_scene.to_string()}"
                    # ip, port = xr_info["addr"]["ip"], xr_info["servicePort"]
                    # self.net_manager.submit_task(
                    #     send_request,
                    #     scene_string,
                    #     f"tcp://{ip}:{port}",
                    #     self.net_manager.zmq_context
                    # )
                    # logger.info(f"The Scene is sent to {xr_info['name']}")
                except Exception as e:
                    logger.error(f"Error when sending scene to xr device: {e}")
                    traceback.print_exc()
            await asyncio_sleep(0.1)

    def send_scene_to_xr_device(self, xr_info: XRNodeInfo):
        send_string_request(
            "SpawnSimScene",
            self.sim_scene.to_string(),
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}"
        )
        if self.sim_scene.root is None:
            logger.warning("The SimScene root is None, nothing to send.")
            return
        self.send_object_to_xr_device(
            xr_info, self.sim_scene.root, self.sim_scene,
        )

    def send_object_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_object: SimObject,
        sim_scene: SimScene,
        parent: Optional[SimObject] = None
    ):
        send_string_request(
            "CreateSimObject",
            sim_object.to_string(sim_scene, parent),
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}"
        )
        for child in sim_object.visuals:
            self.send_visual_to_xr_device(
                xr_info, sim_scene, child, sim_object,
            )
        for child in sim_object.children:
            self.send_object_to_xr_device(
                xr_info, child, sim_scene, sim_object
            )

    def send_visual_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
        sim_visual: SimVisual,
        sim_object: SimObject,
    ):
        send_string_request(
            "CreateVisual",
            sim_visual.to_string(sim_scene, sim_object),
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}"
        )


    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError

# class SimPublisherClient(ServerBase):
#     def __init__(
#         self,
#         sim_scene: SimScene,
#         ip_addr: str = "127.0.0.1",
#         no_rendered_objects: Optional[List[str]] = None,
#         no_tracked_objects: Optional[List[str]] = None,
#         fps: int = 30,
#     ) -> None:
#         self.sim_scene = sim_scene
#         self.fps = fps
#         if no_rendered_objects is None:
#             self.no_rendered_objects = []
#         else:
#             self.no_rendered_objects = no_rendered_objects
#         if no_tracked_objects is None:
#             self.no_tracked_objects = []
#         else:
#             self.no_tracked_objects = no_tracked_objects
#         super().__init__(ip_addr, "SimPublisher")

#     def initialize(self) -> None:
#         self.scene_update_streamer = Streamer(
#             topic_name="SceneUpdate",
#             update_func=self.get_update,
#             fps=self.fps,
#             start_streaming=True)
#         self.asset_service = StrBytesService("Asset", self._on_asset_request)
#         self.xr_device_set: Set[HashIdentifier] = set()
#         self.net_manager.submit_task(self.search_xr_device, self.net_manager)

#     async def search_xr_device(self, node: XRNodeManager):
#         while node.running:
#             for xr_info in node.nodes_info_manager.server_nodes_info.values():
#                 if xr_info["nodeID"] in self.xr_device_set:
#                     continue
#                 if "LoadSimScene" not in xr_info["serviceList"]:
#                     continue
#                 try:
#                     self.xr_device_set.add(xr_info["nodeID"])
#                     scene_string = f"LoadSimScene|{self.sim_scene.to_string()}"
#                     ip, port = xr_info["addr"]["ip"], xr_info["servicePort"]
#                     self.net_manager.submit_task(
#                         send_request,
#                         scene_string,
#                         f"tcp://{ip}:{port}",
#                         self.net_manager.zmq_context
#                     )
#                     logger.info(f"The Scene is sent to {xr_info['name']}")
#                 except Exception as e:
#                     logger.error(f"Error when sending scene to xr device: {e}")
#                     traceback.print_exc()
#             await asyncio_sleep(0.5)

#     def _on_asset_request(self, req: str) -> bytes:
#         return self.sim_scene.raw_data[req]

#     @abc.abstractmethod
#     def get_update(self) -> Dict:
#         raise NotImplementedError
