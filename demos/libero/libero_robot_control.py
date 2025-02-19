"""
Modified from robosuite example scripts.
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.

"""

import argparse
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
import zmq
import threading
from termcolor import colored

from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from libero_example import select_file_from_txt, RobosuitePublisher

from simpub.xr_device.meta_quest3 import MetaQuest3


class MQ3CartController:

    def __init__(self, meta_quest3: MetaQuest3):
        self.meta_quest3 = meta_quest3
        self.last_state = None

    def get_action(self, obs):
        input_data = self.meta_quest3.get_input_data()
        action = np.zeros(7)
        if input_data is None:
            return action
        hand = input_data["right"]
        if self.last_state is not None and hand["hand_trigger"]:
            desired_pos, desired_quat = hand["pos"], hand["rot"]
            last_pos, last_quat = self.last_state
            action[0:3] = (np.array(desired_pos) - np.array(last_pos)) * 100
            d_rot = R.from_quat(desired_quat) * R.from_quat(last_quat).inv()
            action[3:6] = d_rot.as_euler("xyz") * 5
            if hand["index_trigger"]:
                action[-1] = 10
            else:
                action[-1] = -10
            # print(action)
        if not hand["hand_trigger"]:
            self.last_state = None
        else:
            self.last_state = (hand["pos"], hand["rot"])
        return action


class RealRobotJointPDController:

    def __init__(self, real_robot_ip):
        self.pgain = np.array(
            [1000.0, 1000.0, 1000.0, 1000.0, 200.0, 200.0, 300.0]
        )
        self.dgain = np.array([50.0, 50.0, 50.0, 50.0, 6.0, 5.0, 6.0])

        self.sub_socket = zmq.Context().socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{real_robot_ip}:5555")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.data = None
        self.sub_task = threading.Thread(target=self.subscribe_task)
        self.sub_task.start()

    def subscribe_task(self):
        try:
            print('Start to sub')
            while True:
                msg = self.sub_socket.recv_string()
                # print(msg)
                self.data = json.loads(msg)
        except Exception as e:
            print(e)

    def get_action(self, obs):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        joint_pos, joint_vel = obs['robot0_joint_pos'], obs['robot0_joint_vel']
        if self.data is None:
            return np.zeros(8)
        qd_d = self.data['q'] - joint_pos
        vd_d = self.data['dq'] - joint_vel
        action = np.zeros(8)
        action[0:7] = self.pgain * qd_d + self.dgain * vd_d  # original
        if self.data['gripper_width'][0] < 0.9 * self.data['gripper_width'][1]:
            action[-1] = 10
        else:
            action[-1] = -10
        return action


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="meta_quest3")
    parser.add_argument("--task-id", type=int, default=19)
    parser.add_argument("--datasets", type=str, default="libero_90")
    parser.add_argument("--vendor-id", type=int, default=1133)
    parser.add_argument("--product-id", type=int, default=50726)

    args = parser.parse_args()

    if args.device == "real_robot":
        controller_method = "JOINT_POSITION"
    else:
        controller_method = "OSC_POSE"
    controller_config = load_controller_config(
        default_controller=controller_method
    )

    # Create argument configuration
    config = {
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    bddl_file = select_file_from_txt(args.datasets, args.task_id)
    print("Selected task: ", bddl_file)
    assert os.path.exists(bddl_file)
    problem_info = BDDLUtils.get_problem_info(bddl_file)

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    text = colored(language_instruction, "red", attrs=["bold"])
    print("Goal of the following task: ", text)

    if "TwoArm" in problem_name:
        config["env_configuration"] = "single-arm-opposed"
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)
    obs = env.reset()
    publisher = RobosuitePublisher(env, args.host)
    # initialize device
    if args.device == "meta_quest3":
        virtual_controller = MQ3CartController(
            MetaQuest3(device_name="IRLMQ3-1")
        )
    elif args.device == "real_robot":
        virtual_controller = RealRobotJointPDController('141.3.53.152')
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )
    while True:
        obs, _, _, _ = env.step(virtual_controller.get_action(obs))
        env.render()
