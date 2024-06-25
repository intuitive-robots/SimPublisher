import json
from mujoco import mj_name2id, mjtObj
import zmq
import random
from typing import List, Dict
import numpy as np

from simpub.server import SimPublisher, ServerBase
from simpub.parser.mjcf import MJCFParser, MJCFScene
from simpub.simdata import SimObject


class MujocoPublisher(SimPublisher):

    def __init__(self, mj_model, mj_data, mjcf_path: str) -> None:
        self.mj_scene = mj_model
        self.mj_data = mj_data
        self.parser = MJCFParser(mjcf_path)
        self.sim_scene: MJCFScene = self.parser.parse()
        self.tracked_obj_trans: Dict[str, np.ndarray] = dict()
        for child in self.sim_scene.root.children:
            self.tracking_object(child)
        ServerBase.__init__(self)

    def tracking_object(self, obj: SimObject):
        body_id = mj_name2id(self.mj_model, mjtObj.mjOBJ_BODY, obj.name)
        body_jnt_addr = self.mj_model.body_jntadr[body_id]
        pos = self.mj_model.body_pos[body_jnt_addr]
        rot = self.mj_model.body_quat[body_jnt_addr]
        trans = (pos, rot)
        self.tracked_obj_trans[obj.name] = trans
        for child in obj.children:
            self.tracking_object(child)

    def initialize_task(self):
        super().initialize_task()

    def get_update(self) -> Dict[str, List[float]]:
        state = {}
        for name, trans in self.tracked_obj_trans.items():
            pos, rot = trans
            state[name] = [
                -pos[1], pos[2], pos[0], rot[1], -rot[2], -rot[0], rot[3]
            ]
        return state

    def shutdown(self):
        self.discovery_task.shutdown()
        self.stream_task.shutdown()
        self.msg_service.shutdown()

        self.running = False
        self.thread.join()


