from typing import List, Optional, Dict, Tuple
import numpy as np
import taichi as ti

from genesis.engine.scene import Scene as GSScene
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink, RigidGeom

from ..parser.gs import GenesisSceneParser
from ..core.simpub_server import SimPublisher


@ti.data_oriented
class GenesisPublisher(SimPublisher):

    def __init__(
        self,
        gs_scene: GSScene,
        host: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
    ) -> None:
        self.parser = GenesisSceneParser(gs_scene)
        sim_scene = self.parser.sim_scene
        self.update_dict = self.parser.update_dict
        self.tracked_obj_trans: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        super().__init__(
            sim_scene,
            host,
            no_rendered_objects,
            no_tracked_objects,
            fps=60,
        )

    def get_update(self) -> Dict[str, List[float]]:
        # TODO: Fix the problem with taiqi kernel issue
        return {}
        if not self.parser.gs_scene.is_built:
            return {}
        state = {}
        for name, item in self.update_dict.items():
            if name in self.no_tracked_objects:
                continue
            if isinstance(item, RigidEntity):
                pos = item.get_pos().tolist()
                print(pos, item.get_quat())
                quat = item.get_quat().tolist()
            elif isinstance(item, RigidLink):
                pos = item.get_pos().tolist()
                quat = item.get_quat().tolist()
            else:
                continue
            state[name] = [
                pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]
            ]
        return state
