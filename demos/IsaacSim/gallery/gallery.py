# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import time
from pathlib import Path

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
import torch
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sim import UsdFileCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import (
    FRANKA_PANDA_CFG,
    G1_MINIMAL_CFG,
    H1_MINIMAL_CFG,
    KINOVA_GEN3_N7_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_JACO2_N7S300_CFG,
    SAWYER_CFG,
    UR10_CFG,
)
from omni.isaac.lab_assets.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG
from omni.isaac.lab_assets.spot import SPOT_CFG
from omni.isaac.lab_assets.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
)

from simpub.sim.isaacsim_publisher import IsaacSimPublisher


def design_scene() -> tuple[dict, list[list[float]]]:
    scene_entities = {}
    origins = []

    scene_asset_cfg = AssetBaseCfg(
        prim_path="/World/scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{Path(__file__).parent.as_posix()}/simple_room.usd",
        ),
    )
    scene_asset_cfg.spawn.func(scene_asset_cfg.prim_path, scene_asset_cfg.spawn)

    anymal_c_cfg = ANYMAL_C_CFG
    anymal_c_cfg.prim_path = "/World/scene/anymal_c"
    anymal_c_cfg.spawn = None
    scene_entities["anymal_c"] = Articulation(cfg=anymal_c_cfg)
    origins.append([0, 0, 0])

    # return the scene information
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                # # root state
                # root_state = robot.data.default_root_state.clone()
                # root_state[:, :3] += origins[index]
                # robot.write_root_state_to_sim(root_state)
                # joint state
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply default actions to the quadrupedal robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos  # + torch.randn_like(robot.data.joint_pos) * 0.1
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()

    _ = IsaacSimPublisher(
        host="127.0.0.1",
        stage=sim.stage,
        ignored_prim_paths=["/World/defaultGroundPlane"],
        texture_cache_dir=f"{Path(__file__).parent.as_posix()}/.textures",
    )

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
