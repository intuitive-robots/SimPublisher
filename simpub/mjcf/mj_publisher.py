import json
from threading import Thread
from typing import Callable
import mujoco
from mujoco import mj_name2id, mjtObj

from ..unity_data import UnityScene, UnityGameObject, UnityJointType
from simpub.connection.discovery import DiscoverySender
from simpub.connection.streaming import StreamSender
from simpub.connection.service import ReplyService
from simpub.mjcf import MJCFParser

import zmq
import random
import time

import numpy as np


class MjPublisher:
    FPS = 10
    SERVICE_PORT = 5521    # this port is configured in the firewall
    DISCOVERY_PORT = 5520
    STREAMING_PORT = 5522

    def __init__(
        self,
        mj_model,
        mj_data,
        mjcf_path: str,
    ) -> None:

        self.mj_scene = mj_model
        self.mj_data = mj_data

        self.config_zmq()

        self.parser = MJCFParser(mjcf_path)
        self.unity_scene_data = self.parser.parse()
        self.scene_message = self.unity_scene_data.to_string()
        self.track_joints_from_unity_scene(self.unity_scene_data)

    def config_zmq(self):
        self.zmqContext = zmq.Context()
        self.id = random.randint(100_000, 999_999)

        self.running = False
        self.thread = Thread(target=self._loop)
        self.tracked_joints = dict()

    def track_joints_from_unity_scene(self, unity_scene: UnityScene):
        for child in unity_scene.root.children:
            # self.track_joint_from_object(child)
            pass

    def track_joint_from_object(self, obj: UnityGameObject):
        if obj.joint is not None:
            if obj.joint.type == UnityJointType.HINGE:
                mjjoint = self.mj_data.joint(obj.joint.name)
                self.track_joint(obj.joint.name, mjjoint, lambda x: [x.qpos[0]*180/3.14159, x.qvel[0]*180/3.14159])
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
        print(f"free object: {trans}")
        return [-trans[1], trans[2], -trans[0], 0, 0, 0]

    def track_joint(self, joint_name: str, obj, func: Callable):
        print(f"tracking joint: {joint_name}")
        self.tracked_joints[joint_name] = (obj, func)

    def update_joint(self, joint_name: str):
        obj, func = self.tracked_joints[joint_name]
        value = func(obj)
        return value

    def start(self):

        discovery_data = dict()
        discovery_data["SERVICE"] = self.SERVICE_PORT
        discovery_data["STREAMING"] = self.STREAMING_PORT
        discovery_message = f"HDAR:{self.id}:{json.dumps(discovery_data)}"

        # REVIEW: three threads are too much, you can use
        self.service_thread = ReplyService(self.zmqContext, port=self.SERVICE_PORT)
        # REVIEW: Should we use topics for streamer?
        self.streaming_thread = StreamSender(self.zmqContext, port=self.STREAMING_PORT)
        self.discovery_thread = DiscoverySender(discovery_message, self.DISCOVERY_PORT, 2)

        self.service_thread.register_action("SCENE_INFO", self._on_scene_request)
        self.service_thread.register_action("ASSET_DATA", self._on_asset_data_request)

        self.running = True
        self.service_thread.start()
        self.discovery_thread.start()
        self.thread.start()

    def shutdown(self):
        self.discovery_thread.stop()
        self.service_thread.stop()

        self.running = False
        self.thread.join()

    def get_scene(self) -> UnityScene:
        return self.unity_scene_data

    # REVIEW: I suggest to use the register_service method to register the service in a simple way rather than dectator
    # decorator is not necessary, and it is not used in this way
    def register_service(self, tag: str):

        def decorator(func: Callable[[zmq.Socket, str], None]):
            self.service_thread.register_action(tag, func)

        return decorator

    def _on_scene_request(self, socket: zmq.Socket, tag: str):
        socket.send_string(self.scene_message)

    def _on_asset_data_request(self, socket: zmq.Socket, tag: str):
        if tag not in self.unity_scene_data.raw_data:
            print("Received invalid data request")
            return

        socket.send(self.unity_scene_data.raw_data[tag])

    def _loop(self):
        last = 0.0
        print("Starting loop")
        while len(self.tracked_joints) == 0:
            time.sleep(1)
        while self.running:
            print("looping...")
            diff = time.monotonic() - last
            # REVIEW: why not just caculate dt = 1 / self.FPS in advance rather than caculate it every time
            if diff < 1 / self.FPS:
                time.sleep(1 / self.FPS - diff)

            last = time.monotonic()
            msg = {
                "data": {joint_name: self.update_joint(joint_name) for joint_name in self.tracked_joints},
                "time": time.monotonic()
            }
            print(msg)
            self.streaming_thread.publish_dict(msg)
