import mujoco
from mujoco import mj_name2id, mjtObj  # type: ignore
import numpy as np
import time
import os
import argparse
from typing import Optional

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


def check_episode_and_rest(mj_model, mj_data):
    target_ball_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "target_ball")
    object_position = mj_data.xpos[target_ball_id]
    if abs(object_position[0]) > 2 or abs(object_position[1]) > 2:
        reset_ball(mj_model, mj_data)


def reset_ball(mj_model, mj_data):
    target_ball_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "target_ball")
    body_jnt_addr = mj_model.body_jntadr[target_ball_id]
    qposadr = mj_model.jnt_qposadr[body_jnt_addr]
    mj_data.qpos[qposadr:qposadr + 3] = np.array([0, 0, -0.5])
    mj_data.qvel[qposadr:qposadr + 3] = np.array([1.5, 0, 0])


def update_bat(mj_model, mj_data, player1: MetaQuest3, player2: Optional[MetaQuest3] = None):
    bat1_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "bat1")
    player1_input = player1.get_input_data()
    if player1_input is None:
        return
    mj_model.body_pos[bat1_id] = np.array(player1_input["right"]["pos"])
    quat = player1_input["right"]["rot"]
    mj_model.body_quat[bat1_id] = np.array([quat[3], quat[0], quat[1], quat[2]])
    if player2 is not None:
        bat2_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "bat2")
        player2_input = player2.get_input_data()
        if player2_input is None:
            return
        mj_data.body_pos[bat2_id] = np.array(player2_input["left"]["pos"])
        quat = player2_input["left"]["rot"]
        mj_data.body_quat[bat2_id] = np.array([quat[3], quat[0], quat[1], quat[2]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/table_tennis_env.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    last_time = time.time()
    reset_ball(model, data)
    publisher = MujocoPublisher(model, data, args.host)
    player1 = MetaQuest3("IRLMQ3-1")
    # player2 = MetaQuest3("ALRMQ3-1")
    # player2 = MetaQuest3("ALR2")
    # while not player1.connected:
    #     time.sleep(0.01)
    count = 0
    while True:
        try:
            mujoco.mj_step(model, data)
            if time.time() - last_time < 0.002:
                time.sleep(0.002 - (time.time() - last_time))
            last_time = time.time()
            update_bat(model, data, player1)
            if count % 10 == 0:
                check_episode_and_rest(model, data)
            count += 1
        except KeyboardInterrupt:
            break
    publisher.shutdown()
