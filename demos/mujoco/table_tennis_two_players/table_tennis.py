import mujoco
from mujoco import mj_name2id, mjtObj  # type: ignore
import numpy as np
import time
import os

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


def check_episode_and_rest(mj_model, mj_data):
    target_ball_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "target_ball")
    object_position = mj_data.xpos[target_ball_id]
    if abs(object_position[0]) > 2 or abs(object_position[1]) > 2:
        body_jnt_addr = mj_model.body_jntadr[target_ball_id]
        qposadr = mj_model.jnt_qposadr[body_jnt_addr]
        mj_data.qpos[qposadr:qposadr + 3] = np.array([0, 0, 2])
        mj_data.qvel[qposadr:qposadr + 3] = np.array([0, 0, 0])


def update_bat(mj_model, mj_data, player1: MetaQuest3, player2: MetaQuest3 = None):
    bat1_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "bat1")
    player1_input = player1.get_input_data()
    # print(player1_input)
    if player1_input is None:
        return
    mj_model.body_pos[bat1_id] = np.array(player1_input["right"]["pos"])
    quat = player1_input["right"]["rot"]
    mj_model.body_quat[bat1_id] = np.array([quat[3], quat[0], quat[1], quat[2]])
    # bat2_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "bat2")
    # mj_data.body_pos[bat2_id] = np.array(player1.input_data["left"]["pos"])
    # mj_data.body_quat[bat2_id] = np.array(player1.input_data["left"]["rot"])


if __name__ == '__main__':
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/table_tennis_env.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    last_time = time.time()
    publisher = MujocoPublisher(model, data)
    # player1 = MetaQuest3("UnityClient")
    # # player2 = MetaQuest3("ALR2")
    # while not player1.connected:
    #     time.sleep(0.01)
    count = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
        if time.time() - last_time < 0.001:
            time.sleep(0.001 - (time.time() - last_time))
        last_time = time.time()
        # if count % 10 == 0:
        #     update_bat(model, data, player1)
        #     check_episode_and_rest(model, data)
        count += 1
    publisher.shutdown()
