from __future__ import annotations

import abc
import traceback
from asyncio import sleep as asyncio_sleep
from typing import Dict, List, Optional, Set, TypedDict
import concurrent.futures

import pyzlc
from pyzlc.sockets.publisher import Streamer

from .log import func_timing, logger
from ..parser.simdata import TreeNode, SimScene
from ..simpubweb.simpub_web_server import SimPubWebServer

from .utils import (
    HashIdentifier,
    XRNodeInfo
)


class ServerBase(abc.ABC):
    def __init__(self, server_name: str, ip_addr: str):
        self.ip_addr: str = ip_addr
        pyzlc.init(server_name, ip_addr, group="239.255.10.10", group_port=7720, group_name="IRIS")
        assert pyzlc.LanComNode.instance is not None, "LanComNode is not initialized."
        self.node_manager = pyzlc.LanComNode.instance
        self.web_server: Optional[SimPubWebServer] = None
        self.web_server_future: Optional[concurrent.futures.Future] = None
        self._start_web_server()
        self.initialize()

    def _start_web_server(self):
        if self.web_server is not None:
            return
        try:
            self.web_server = SimPubWebServer(host="127.0.0.1", port=5000)
            self.web_server_future = pyzlc.submit_thread_pool_task(
                self.web_server.serve_forever
            )
            logger.info("Web dashboard is running at http://127.0.0.1:5000")
        except Exception as e:
            logger.error(
                f"Failed to start web dashboard on 127.0.0.1:5000: {e}"
            )
            traceback.print_exc()


    def spin(self):
        pyzlc.spin()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError


class MsgServer(ServerBase):
    def initialize(self):
        pass


class RigidObjectUpdateData(TypedDict):
    data: Dict[str, List[float]]


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
            self.sim_scene.config,
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
        await self.send_objects_to_xr_device(
            self.sim_scene.root,
            scene_prefix,
        )
        await self.send_lights_to_xr_device(scene_prefix)

    async def send_objects_to_xr_device(self, sim_object_node: TreeNode, scene_prefix: str):
        # print(f"Sending object {sim_object_node.data['name']} to xr device")
        assert sim_object_node.data is not None, "SimObject node data is None"
        await pyzlc.async_call(
            f"{scene_prefix}/CreateSimObject",
            sim_object_node.data,
        )
        for child in sim_object_node.children:
            await self.send_objects_to_xr_device(child, scene_prefix)

    async def send_lights_to_xr_device(self, scene_prefix: str):
        for light in self.sim_scene.lights:
            await pyzlc.async_call(
                f"{scene_prefix}/CreateLight",
                light,
            )


    @abc.abstractmethod
    def get_update(self) -> RigidObjectUpdateData:
        raise NotImplementedError
