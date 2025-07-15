from __future__ import annotations
import abc
from typing import Dict, List, Optional, Set
from asyncio import sleep as asyncio_sleep
import traceback

from ..parser.simdata import SimScene, SimObject
from .net_manager import XRNodeManager, init_xr_node_manager, Streamer
from .log import logger, func_timing
from .utils import (
    send_raw_request_async,
    HashIdentifier,
    XRNodeInfo,
    get_zmq_socket_url,
)


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
        fps: int = 45,
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
        self.scene_update_streamer = Streamer(
            topic_name="RigidObjectUpdate",
            update_func=self.get_update,
            fps=self.fps,
            start_streaming=True,
        )
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
                    await self.send_scene_to_xr_device(
                        node_manager.xr_nodes_info[xr_node_id]
                    )
                except Exception as e:
                    logger.error(f"Error when sending scene to xr device: {e}")
                    traceback.print_exc()
            await asyncio_sleep(0.1)

    @func_timing
    async def send_scene_to_xr_device(self, xr_info: XRNodeInfo):
        await send_raw_request_async(
            [
                "DeleteSimScene".encode(),
                self.sim_scene.name.encode(),
            ],
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}",
        )
        await send_raw_request_async(
            [
                "SpawnSimScene".encode(),
                self.sim_scene.serialize().encode(),
            ],
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}",
        )
        if self.sim_scene.root is None:
            logger.warning("The SimScene root is None, nothing to send.")
            return
        await self.send_rigid_body_streamer(
            xr_info,
            self.sim_scene,
        )
        await self.send_objects_to_xr_device(
            xr_info,
            self.sim_scene,
            self.sim_scene.root,
        )
        await self.send_assets_to_xr_device(
            xr_info,
            self.sim_scene,
        )

    async def send_objects_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
        sim_object: SimObject,
        parent: Optional[SimObject] = None,
    ):
        await send_raw_request_async(
            [
                f"{sim_scene.name}/CreateSimObject".encode(),
                parent.name.encode() if parent else "".encode(),
                sim_object.serialize().encode(),
            ],
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}",
        )

    async def send_assets_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
    ):
        if sim_scene.root is None:
            logger.warning("The SimScene root is None, nothing to send.")
            return
        for (
            name,
            sim_visual,
            mesh_raw_data,
            texture_raw_data,
        ) in sim_scene.get_all_assets(sim_scene.root):
            await send_raw_request_async(
                [
                    f"{sim_scene.name}/CreateVisual".encode(),
                    name.encode(),
                    sim_visual.serialize().encode(),
                    mesh_raw_data,
                    texture_raw_data,
                ],
                f"tcp://{xr_info['ip']}:{xr_info['servicePort']}",
            )

    async def send_rigid_body_streamer(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
    ):
        url = get_zmq_socket_url(self.scene_update_streamer.socket)
        await send_raw_request_async(
            [
                f"{sim_scene.name}/SubscribeRigidObjectsController".encode(),
                url.encode(),
                "RigidObjectUpdate".encode(),
            ],
            f"tcp://{xr_info['ip']}:{xr_info['servicePort']}",
        )

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
