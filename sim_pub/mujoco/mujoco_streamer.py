from websockets import WebSocketServerProtocol
from sim_pub.base import ObjPublisherBase

from ..primitive import SimStreamer
from ..model_loader import SceneLoader

class MujocoStreamer(SimStreamer):
    def __init__(self, model_path: str) -> None:
        self.scene_loader = SceneLoader()
        self.scene_loader.include_mjcf_file(model_path)
        scene_msg = self.scene_loader.generate_scene_msg()
        super().__init__()
        
    async def on_connect(self, ws: WebSocketServerProtocol) -> None:
        await super().on_connect(ws)
        scene_msg = self.scene_loader.generate_scene_msg()
        await self._send_msg_on_loop(scene_msg, ws)
