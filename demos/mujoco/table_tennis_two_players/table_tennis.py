import mujoco
from mujoco import mj_name2id, mjtObj  # type: ignore
import numpy as np
import time
import os
import argparse
from scipy.spatial.transform import Rotation

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


def check_episode_and_rest(mj_model, mj_data):
    target_ball_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "target_ball")
    object_position = mj_data.xpos[target_ball_id]
    if abs(object_position[0]) > 2 or abs(object_position[1]) > 2:
        reset_ball(mj_model, mj_data)


def reset_ball(mj_model, mj_data):
    print("Resetting ball")
    target_ball_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "target_ball")
    body_jnt_addr = mj_model.body_jntadr[target_ball_id]
    qposadr = mj_model.jnt_qposadr[body_jnt_addr]
    mj_data.qpos[qposadr : qposadr + 3] = np.array([0, 0, -0.3])
    mj_data.qvel[qposadr : qposadr + 3] = np.array([1.5, 0, 0])


def update_bat(mj_model, mj_data, player: MetaQuest3, bat_name, hand="right"):
    """
    Update a bat in MuJoCo using velocity control to track XR controller position and orientation.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        player: MetaQuest3 object with controller data
        bat_name: Name of the bat body in the MuJoCo model
        hand: Which hand controller to use ("right" or "left")
    """
    # Control parameters - tune these for responsiveness vs. stability
    pos_gain = 100.0  # Position tracking gain
    rot_gain = 0.5  # Rotation tracking gain
    max_vel = 100.0  # Maximum linear velocity
    max_angvel = 40.0  # Maximum angular velocity
    # Get bat body ID
    bat_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY, bat_name)
    if bat_id < 0:
        print(f"Warning: Could not find body named '{bat_name}'")
        return
    # Find the joint that controls this bat
    bat_joint_id = None
    for j in range(mj_model.njnt):
        if mj_model.jnt_bodyid[j] == bat_id:
            bat_joint_id = j
            break
    if bat_joint_id is None:
        print(f"Warning: Could not find joint for '{bat_name}'")
        return
    bat_dofadr = mj_model.jnt_dofadr[bat_joint_id]
    # Get player input
    player_input = player.get_input_data()
    if player_input is None or hand not in player_input:
        mj_data.qvel[bat_dofadr : bat_dofadr + 3] = np.array([0, 0, 0])
        # Angular velocity - next 3 DOFs
        mj_data.qvel[bat_dofadr + 3 : bat_dofadr + 6] = np.array([0, 0, 0])
        return
    # Get target position and orientation from controller
    target_pos = np.array(player_input[hand]["pos"])
    quat = player_input[hand]["rot"]
    # Convert to MuJoCo quaternion format (w, x, y, z)
    target_quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    # Get current object position and orientation directly from MuJoCo
    current_pos = mj_data.xpos[bat_id].copy()
    current_quat = mj_data.xquat[bat_id].copy()
    # Calculate position error and desired velocity
    pos_error = target_pos - current_pos
    desired_vel = pos_gain * pos_error
    # Limit maximum velocity
    vel_norm = np.linalg.norm(desired_vel)
    if vel_norm > max_vel:
        desired_vel = desired_vel * (max_vel / vel_norm)
    desired_angular_vel = rot_gain * get_rot_vec_error(
        target_quat, current_quat
    )
    # Limit maximum angular velocity
    ang_vel_norm = np.linalg.norm(desired_angular_vel)
    if ang_vel_norm > max_angvel:
        desired_angular_vel = desired_angular_vel * (max_angvel / ang_vel_norm)
    mj_data.qvel[bat_dofadr : bat_dofadr + 3] = desired_vel
    mj_data.qvel[bat_dofadr + 3 : bat_dofadr + 6] = desired_angular_vel


def get_rot_vec_error(target_quat, current_quat):
    """
    Calculate the rotation vector error between target and current quaternions.

    Args:
        target_quat: Target quaternion in [w, x, y, z] format
        current_quat: Current quaternion in [w, x, y, z] format
    """
    # Convert from [w, x, y, z] to SciPy's [x, y, z, w] format
    target_scipy_quat = np.array(
        [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
    )
    current_scipy_quat = np.array(
        [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
    )

    # Create Rotation objects
    target_rot = Rotation.from_quat(target_scipy_quat)
    current_rot = Rotation.from_quat(current_scipy_quat)
    rot_error = current_rot.inv() * target_rot
    error = rot_error.as_rotvec(degrees=True)
    # error = target_rot.as_rotvec(degrees=True) - current_rot.as_rotvec(degrees=True)
    # for i in range(3):
    #     if error[i] > 180:
    #         error[i] -= 360
    #     if error[i] < -180:
    #         error[i] += 360
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/table_tennis_env.xml",
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    last_time = time.time()
    publisher = MujocoPublisher(model, data, args.host)
    player1 = MetaQuest3("IRLMQ3-2")
    # player1 = MetaQuest3("UnityNode")
    # # player2 = MetaQuest3("ALR2")
    # while not player1.connected:
    #     time.sleep(0.01)
    count = 0
    reset_ball(model, data)
    while True:
        try:
            mujoco.mj_step(model, data)
            interval = time.time() - last_time
            if interval < 0.002:
                time.sleep(0.002 - interval)
            last_time = time.time()
            update_bat(model, data, player1, "bat1")
            # if count % 10 == 0:
            check_episode_and_rest(model, data)
            # count += 1
        except KeyboardInterrupt:
            break
    publisher.shutdown()
