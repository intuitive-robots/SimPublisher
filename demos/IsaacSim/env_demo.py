# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import torch
import carb

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
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from simpub.sim.isaacsim_publisher import IsaacSimPublisher

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

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
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
