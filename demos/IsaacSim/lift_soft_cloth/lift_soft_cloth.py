# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to demonstrate lifting a deformable object with a robotic arm.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_teddy_bear.py

"""

"""Launch Omniverse Toolkit first."""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift a teddy bear with a robotic arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Callable
from dataclasses import MISSING

import gymnasium as gym
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import torch
import trimesh
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.sim.spawners.meshes import meshes
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import FrankaTeddyBearLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from simpub.sim.isaacsim_publisher import IsaacSimPublisher


class FrankaSoftBasketLiftEnvCfg(FrankaTeddyBearLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        self.events.reset_object_position = None

        self.scene.table = None
        self.scene.plane.init_state.pos = [0, 0, 0]

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.85, 0.1, 0.5), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{Path(__file__).parent.as_posix()}/jacket.usd",
                scale=(0.75, 0.75, 0.75),
                # deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                # physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            ),
        )

        self.scene.prop_0 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prop0",
            spawn=sim_utils.MeshCuboidCfg(
                size=(1.5, 0.6, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=20.0, dynamic_friction=20.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.1, 0.2)),
            debug_vis=True,
        )


gym.register(
    id="Isaac-Lift-Soft-Basket-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:FrankaSoftBasketLiftEnvCfg",
    },
    disable_env_checker=True,
)


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Soft-Basket-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    env_cfg.viewer.eye = (2.1, 1.0, 1.3)

    # create environment
    env = gym.make("Isaac-Lift-Soft-Basket-Franka-IK-Abs-v0", cfg=env_cfg).unwrapped
    # reset environment at start
    env.reset()

    # start simpub server
    if env.sim is not None and env.sim.stage is not None:
        print("parsing usd stage...")
        # publisher = IsaacSimPublisher(host="127.0.0.1", stage=env.unwrapped.sim.stage)
        publisher = IsaacSimPublisher(host="127.0.0.1", stage=env.sim.stage)

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)

    ee_poses = torch.tensor(
        [
            [0.2, 0.03, 0.82, 0.5609855, 0.4304593, 0.4304593, 0.5609855, 1.0],
            [0.45, 0.025, 0.35, 0, 0, 1, 0, 1.0],
            [0.45, 0.025, 0.35, 0, -0.258819, 0.9659258, 0, 1.0],
            [0.45, 0.025, 0.21, 0, -0.258819, 0.9659258, 0, 1.0],
            [0.45, 0.025, 0.21, 0, -0.258819, 0.9659258, 0, -1.0],
            [0.45, 0.025, 0.21, 0, -0.258819, 0.9659258, 0, -1.0],
            [0.6, -0.15, 0.21, 0, -0.258819, 0.9659258, 0, -1.0],
        ]
    )
    actions = ee_poses[0, :].reshape(1, -1)

    time_step = env_cfg.sim.dt * env_cfg.decimation
    time_elapsed = 0
    i_action = 0

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            if time_elapsed >= 1.0:
                i_action = np.clip(i_action + 1, 0, len(ee_poses) - 1)
                actions = ee_poses[i_action, :].reshape(1, -1)
                time_elapsed = 0.0

            if torch.any(dones):
                i_action = 0
                actions = ee_poses[i_action, :].reshape(1, -1)
                time_elapsed = 0.0

            time_elapsed += time_step

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
