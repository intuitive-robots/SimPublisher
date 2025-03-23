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
from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.spawners.meshes import meshes
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import FrankaTeddyBearLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from simpub.sim.isaacsim_publisher import IsaacSimPublisher


@clone
def _spaw_func(
    prim_path: str,
    cfg: "MeshTrimeshCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    mesh_obj: trimesh.Trimesh = trimesh.load(cfg.mesh)
    mesh_obj.fix_normals()
    # mesh_obj.apply_transform(trimesh.transformations.scale_matrix(0.75))
    mesh_obj.apply_transform(trimesh.transformations.euler_matrix(math.pi * 0.5, 0, math.pi * 0.5))
    meshes._spawn_mesh_geom_from_mesh(
        prim_path=prim_path,
        cfg=cfg,
        mesh=mesh_obj,
        translation=translation,
        orientation=orientation,
    )
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class MeshTrimeshCfg(MeshCfg):
    """Configuration parameters for a sphere mesh prim with deformable properties.

    See :meth:`spawn_mesh_sphere` for more information.
    """

    func: Callable = _spaw_func

    mesh: str = MISSING


class FrankaSoftBasketLiftEnvCfg(FrankaTeddyBearLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0

        self.events.reset_object_position = None

        self.scene.table = None
        self.scene.plane.init_state.pos = [0, 0, 0]

        self.scene.robot_2 = copy.deepcopy(self.scene.robot)
        self.scene.robot_2.prim_path = "{ENV_REGEX_NS}/Robot2"
        self.scene.robot_2.init_state.pos = [1.4, 0, 0]
        self.scene.robot_2.init_state.rot = [0, 0, 0, 1]

        self.actions.arm_action_2 = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action_2 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.7, 0, 0.5), rot=(0.707, 0, 0, 0.707)),
            spawn=MeshTrimeshCfg(
                mesh=f"{Path(__file__).parent.as_posix()}/basket.obj",
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    poissons_ratio=0.4, youngs_modulus=5e2, dynamic_friction=100.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
        )

        self.scene.prop_0 = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prop0",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.4, 0.4, 0.2),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e8),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.7, 0, 0.2)),
            debug_vis=True,
        )

        self.scene.prop_1 = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prop1",
            spawn=sim_utils.MeshSphereCfg(
                radius=0.1,
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e2),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.7, -0.1, 1.0)),
            debug_vis=True,
        )

        self.scene.prop_2 = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prop2",
            spawn=sim_utils.MeshSphereCfg(
                radius=0.1,
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.3)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e2),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.7, -0.1, 1.2)),
            debug_vis=True,
        )

        self.scene.prop_3 = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prop3",
            spawn=sim_utils.MeshSphereCfg(
                radius=0.1,
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.7, 0.3)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e2),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.7, -0.1, 1.4)),
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
    actions = torch.zeros(env.action_space.shape, device=env.device)

    # fmt: off
    ee_poses = torch.tensor(
        [
            [0.40, 0.03, 0.62, 0.5609855, 0.4304593, 0.4304593, 0.5609855, 1.0] * 2,
            [0.53, 0.03, 0.62, 0.5609855, 0.4304593, 0.4304593, 0.5609855, 1.0] * 2,
            [0.53, 0.03, 0.62, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.90, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.70, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.90, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.70, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.90, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 0.70, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            [0.53, 0.03, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0] * 2,
            
            [0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
            [0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
            [0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
            [0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
            [0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
            [0.53, -0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0, 0.53, 0.15, 1.00, 0.5609855, 0.4304593, 0.4304593, 0.5609855, -1.0],
        ]
    )
    # fmt: on
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
