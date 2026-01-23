from __future__ import annotations

import abc
import traceback
from asyncio import sleep as asyncio_sleep
from typing import Dict, List, Optional, Set

import pyzlc
from pyzlc.sockets.publisher import Streamer

from .log import func_timing, logger
from ..parser.simdata import TreeNode, SimScene


from .utils import (
    HashIdentifier,
    XRNodeInfo
)


class ServerBase(abc.ABC):
    def __init__(self, server_name: str, ip_addr: str):
        self.ip_addr: str = ip_addr
        pyzlc.init(server_name, ip_addr)
        assert pyzlc.LanComNode.instance is not None, "LanComNode is not initialized."
        self.node_manager = pyzlc.LanComNode.instance
        self.initialize()

    def spin(self):
        pyzlc.spin()

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
        super().__init__(sim_scene.name, ip_addr)

    def initialize(self) -> None:
        self.scene_update_streamer = Streamer(
            topic_name="RigidObjectUpdate",
            update_func=self.get_update,
            fps=self.fps,
            start_streaming=True,
        )
        self.xr_device_set: Set[HashIdentifier] = set()
        pyzlc.submit_loop_task(self.search_xr_device())

    async def search_xr_device(self):
        # TODO: If Unity restarts too quickly (under 1 second),
        # TODO: the scene may not be fully constructed when
        # TODO: SimPublisher attempts to send it.
        # TODO: In such cases, this function may fail
        # TODO: and require a SimPublisher restart.
        while pyzlc.is_running():
            for xr_info in pyzlc.get_nodes_info():
                if not xr_info["name"].startswith("IRIS/Device/"):
                    continue
                if xr_info["nodeID"] in self.xr_device_set:
                    continue
                self.xr_device_set.add(xr_info["nodeID"])
                try:
                    await self.send_scene_to_xr_device(xr_info)
                except Exception as e:
                    logger.error(f"Error when sending scene to xr device: {e}")
                    traceback.print_exc()
            await asyncio_sleep(0.5)

    @func_timing
    async def send_scene_to_xr_device(self, xr_info: XRNodeInfo):
        logger.info(f"Sending scene to xr device: {xr_info['name']}")
        node_prefix = f"{xr_info['name']}"
        await pyzlc.wait_for_service_async(f"{node_prefix}/DeleteSimScene", timeout=5.0)
        await pyzlc.async_call(
            f"{node_prefix}/DeleteSimScene",
            self.sim_scene.name,
        )
        await pyzlc.async_call(
            f"{node_prefix}/SpawnSimScene",
            self.sim_scene.setting,
        )
        if self.sim_scene.root is None:
            logger.warning("The SimScene root is None, nothing to send.")
            return
        scene_prefix = f"{node_prefix}/{self.sim_scene.name}"
        await pyzlc.wait_for_service_async(
            f"{scene_prefix}/SubscribeRigidObjectsController", timeout=1.0
        )
        await pyzlc.async_call(
            f"{scene_prefix}/SubscribeRigidObjectsController",
            self.scene_update_streamer.url,
        )
        # print("Subscribed RigidObjectsController", self.scene_update_streamer.url)
        await self.send_objects_to_xr_device(
            self.sim_scene.root,
            scene_prefix,
        )
        # await self.send_objects_to_xr_device(
        #     xr_info,
        #     self.sim_scene,
        #     self.sim_scene.root,
        # )
        # await self.send_assets_to_xr_device(
        #     xr_info,
        #     self.sim_scene,
        # )

    async def send_objects_to_xr_device(self, sim_object_node: TreeNode, scene_prefix: str):
        # print(f"Sending object {sim_object_node.data['name']} to xr device")
        assert sim_object_node.data is not None, "SimObject node data is None"
        await pyzlc.async_call(
            f"{scene_prefix}/CreateSimObject",
            sim_object_node.data,
        )
        for child in sim_object_node.children:
            await self.send_objects_to_xr_device(child, scene_prefix)

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
