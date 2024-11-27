# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import torch
import carb
from scipy.spatial.transform import Rotation

from omni.isaac.lab.app import AppLauncher
import gymnasium as gym

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# isaac sim modules are only available after launching it (?)

from pxr import Usd

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObjectCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.spawners.shapes.shapes_cfg import CapsuleCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
#     DifferentialInverseKinematicsActionCfg,
# )

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from simpub.sim.isaacsim_publisher import IsaacSimPublisher
# from simpub.xr_device.meta_quest3 import MetaQuest3

print(ISAACLAB_NUCLEUS_DIR)


def pre_process_actions(
    task: str, delta_pose: torch.Tensor, gripper_command: bool
) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    task = "Isaac-Lift-Cube-Franka-IK-Rel-v0"

    # parse configuration
    # why using undefined parameter (use_gpu) doesn't raise error?
    env_cfg = parse_env_cfg(
        task_name=task,
        num_envs=1,
        use_fabric=True,
    )

    # add some primitives to the env
    env_cfg.scene.cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.7, 0, 0.155], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    env_cfg.scene.cube3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Capsule1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0, 0.055], rot=[1, 0, 0, 0]
        ),
        spawn=CapsuleCfg(
            radius=0.03,
            height=0.1,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 1.0)),
        ),
    )

    # # you should probably set use_relative_mode to False when using meta quest 3 input
    # # but currently it does not work (seems to be a problem of coordiante system alignment)
    # env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["panda_joint.*"],
    #     body_name="panda_hand",
    #     controller=DifferentialIKControllerCfg(
    #         command_type="pose", use_relative_mode=False, ik_method="dls"
    #     ),
    #     scale=1.0,
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
    #         pos=[0.0, 0.0, 0.107]
    #     ),
    # )

    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(task, cfg=env_cfg)
    # print(env.sim)

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in task:
        carb.log_warn(
            f"The environment '{task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    sensitivity = 10
    teleop_interface = Se3Keyboard(
        pos_sensitivity=0.005 * sensitivity,
        rot_sensitivity=0.005 * sensitivity,
    )

    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("simulation context:", env.sim)
    print("stage:", env.sim.stage)

    if env.sim is not None and env.sim.stage is not None:
        print("parsing usd stage...")
        publisher = IsaacSimPublisher(host="192.168.0.134", stage=env.sim.stage)
        # publisher = IsaacSimPublisher(host="127.0.0.1", stage=env.sim.stage)

    # meta_quest3 = MetaQuest3("ALRMetaQuest3")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # # get meta quest 3 input (does not work...)
            # input_data = meta_quest3.get_input_data()
            # if input_data is None:
            #     continue

            # input_pos = input_data["right"]["pos"]
            # input_pos[2] -= 0.1
            # input_rot = input_data["right"]["rot"]
            # input_rot = Rotation.from_quat(input_rot).as_quat()
            # input_gripper = input_data["right"]["index_trigger"]

            # input_pos = [0.4, 0.0, 0.2]
            # input_rot = Rotation.from_euler("XYZ", [0, 180, 0], degrees=True).as_quat()

            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()

            # # get command from meta quest 3
            # delta_pose = np.array(
            #     [
            #         input_pos[0],
            #         input_pos[1],
            #         input_pos[2],
            #         input_rot[3],
            #         input_rot[0],
            #         input_rot[1],
            #         input_rot[2],
            #     ]
            # )
            # print(delta_pose)
            # gripper_command = input_gripper

            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )
            # pre-process actions
            actions = pre_process_actions(task, delta_pose, gripper_command)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
