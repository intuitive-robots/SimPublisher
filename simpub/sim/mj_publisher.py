import json
from typing import Callable
from mujoco import mj_name2id, mjtObj
import zmq
import random
from typing import List

from simpub.server import SimPublisher, TaskBase, PortSet
from simpub.server import MsgService, StreamTask, DiscoveryTask
from simpub.parser.mjcf import MJCFParser, MJCFScene
from simpub.scenestruct.unity import UnityScene, SimObject
from simpub.scenestruct.unity import UnityJointType


class MujocoPublisher(SimPublisher):

    def __init__(self, mj_model, mj_data, mjcf_path: str) -> None:
        self.mj_scene = mj_model
        self.mj_data = mj_data
        self.parser = MJCFParser(mjcf_path)
        self.mjcf_scene: MJCFScene = self.parser.parse()
        self.scene_message = self.mjcf_scene.to_string()
        for child in self.mjcf_scene.root.children:
            self.track_joint_from_object(child)
        super().__init__()

    def setup_zmq(self):
        self.zmqContext = zmq.Context()
        self.id = random.randint(100_000, 999_999)
        self.tracked_joints = dict()

    def tracking_object(self, obj: SimObject):
        
        if obj.joint is not None:
            if obj.joint.type == UnityJointType.HINGE:
                mjjoint = self.mj_data.joint(obj.joint.name)
                self.track_joint(obj.joint.name, mjjoint, lambda x: [-x.qpos[0]*180/3.14159, -x.qvel[0]*180/3.14159])
            if obj.joint.type == UnityJointType.SLIDE:
                mjjoint = self.mj_data.joint(obj.joint.name)
                self.track_joint(obj.joint.name, mjjoint, lambda x: [x.qpos[0], x.qvel[0]])
            elif obj.joint.type == UnityJointType.FREE:
                body_id = mj_name2id(self.mj_model, mjtObj.mjOBJ_BODY, obj.name)
                body_jnt_addr = self.mj_model.body_jntadr[body_id]
                qposadr = self.mj_model.jnt_qposadr[body_jnt_addr]
                trans = self.mj_data.qpos[qposadr: qposadr + 3]
                self.track_joint(obj.joint.name, trans, lambda trans: [-trans[1], trans[2], trans[0], 0, 0, 0])

        for child in obj.children:
            self.track_joint_from_object(child)

    def get_free_object_data(self, qposadr):
        trans = self.mj_data.qpos[qposadr: qposadr + 3]
        return [-trans[1], trans[2], -trans[0], 0, 0, 0]

    def track_joint(self, joint_name: str, obj, func: Callable):
        self.tracked_joints[joint_name] = (obj, func)

    def update_joint(self, joint_name: str):
        obj, func = self.tracked_joints[joint_name]
        value = func(obj)
        return value

    def initialize_task(self):
        self.tasks: List[TaskBase] = []
        discovery_data = {
            "SERVICE": PortSet.SERVICE,
            "STREAMING": PortSet.STREAMING,
        }
        discovery_message = f"SimPub:{self.id}:{json.dumps(discovery_data)}"
        self.discovery_task = DiscoveryTask(self.zmqContext, discovery_message)
        self.tasks.append(self.discovery_task)

        self.stream_task = StreamTask(self, self.zmqContext)
        self.tasks.append(self.stream_task)

        self.msg_service = MsgService(self.zmqContext)
        self.msg_service.register_action("SCENE", self._on_scene_request)
        self.msg_service.register_action("ASSET", self._on_asset_request)

    def shutdown(self):
        self.discovery_task.shutdown()
        self.stream_task.shutdown()
        self.msg_service.shutdown()

        self.running = False
        self.thread.join()

    def _on_scene_request(self, socket: zmq.Socket, tag: str):
        socket.send_string(self.scene_message)

    def _on_asset_request(self, socket: zmq.Socket, tag: str):
        if tag not in self.mj_scene.raw_data:
            print("Received invalid data request")
            return

        socket.send(self.mj_scene.raw_data[tag])
