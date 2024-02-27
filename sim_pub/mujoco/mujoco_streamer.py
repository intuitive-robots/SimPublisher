from sim_pub.base import ObjPublisherBase

from ..primitive import SimStreamer
from ..model_loader import SceneLoader

class MujocoStreamer(SimStreamer):
    def __init__(self, model_path: str) -> None:
        scene_loader = SceneLoader()
        scene_loader.include_mjcf_file(model_path)
        
        super().__init__(dt, publisher_list, host, port, on_stream)