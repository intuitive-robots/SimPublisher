from __future__ import annotations
import abc
from typing import Dict, List
from urllib.parse import parse_qs
import time
import aiohttp.web

from simpub.simdata import SimScene
from .net_manager import init_net_manager
from .net_manager import Streamer, Service
from .net_manager import ServerPort
from .log import logger


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


# class SimPubHTTPServer(HTTPServer):

#     def __init__(self, server_address, RequestHandlerClass):
#         self.asset_data = {}
#         super().__init__(server_address, RequestHandlerClass)

#     def upload_asset_data(self, asset_tag, asset_data: bytes):
#         self.asset_data[asset_tag] = asset_data

class SimPubHTTPServer:

    def __init__(self, host: str):
        self.asset_data = {}
        self.host = host
        self.app = aiohttp.web.Application()
        self.app.router.add_get('/', self.handle_get)

    async def handle_get(self, request):
        query_params = parse_qs(request.query_string)
        asset_tag = query_params.get("asset_tag", [None])[0]
        asset_tag = query_params.get("asset_tag", [None])[0]
        start_time = time.time()
        logger.info(f"Received GET request for asset: {asset_tag} at {start_time}")
        mesh_data = self.retrieve_asset_data(asset_tag)
        if mesh_data is None:
            return aiohttp.web.Response(status=404, text="Asset not found")
        return aiohttp.web.Response(body=mesh_data, content_type="application/octet-stream")
        logger.info(f"Sent asset: {asset_tag} in {time.time() - start_time:.2f}s")

    def upload_asset_data(self, asset_tag, asset_data: bytes):
        self.asset_data[asset_tag] = asset_data

    def retrieve_asset_data(self, asset_tag):
        return self.asset_data.get(asset_tag)

    async def start_aiohttp_server(self):
        runner = aiohttp.web.AppRunner(self.app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, host=self.host, port=ServerPort.HTTP)
        await site.start()


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

    def initialize(self):
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        # self.http_server = SimPubHTTPServer(
        #     (self.host, ServerPort.HTTP), AssetHTTPHandler
        # )
        # self.http_server_thread = self.net_manager.executor.submit(
        #     self.http_server.serve_forever
        # )
        self.http_server = SimPubHTTPServer(self.host)
        self.http_server_task = self.net_manager.submit_task(
            self.http_server.start_aiohttp_server
        )
        # load the asset to http server
        for tag, raw_data in self.sim_scene.raw_data.items():
            self.http_server.upload_asset_data(tag, raw_data)
        self.scene_service = Service("Scene", self._on_scene_request, str)
        # self.asset_service = Service("Asset", self._on_asset_request, bytes)

    def _on_scene_request(self, req: str) -> str:
        return self.sim_scene.to_string()

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
