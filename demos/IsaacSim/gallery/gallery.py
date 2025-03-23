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

from isaaclab.app import AppLauncher

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
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import (
    FRANKA_PANDA_CFG,
    G1_MINIMAL_CFG,
    H1_CFG,
    H1_MINIMAL_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
    UR10_CFG,
)
from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
)

from simpub.sim.isaacsim_publisher import IsaacSimPublisher


def design_scene():
    scene_entities = {}

    scene_asset_cfg = AssetBaseCfg(
        prim_path="/World/scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{Path(__file__).parent.as_posix()}/gallery.usd",
        ),
    )
    scene_asset_cfg.spawn.func(scene_asset_cfg.prim_path, scene_asset_cfg.spawn)

    anymal_b_cfg = ANYMAL_B_CFG
    anymal_b_cfg.prim_path = "/World/scene/anymal_b"
    anymal_b_cfg.spawn = None
    scene_entities["anymal_b"] = Articulation(cfg=anymal_b_cfg)

    anymal_d_cfg = ANYMAL_D_CFG
    anymal_d_cfg.prim_path = "/World/scene/anymal_d"
    anymal_d_cfg.spawn = None
    scene_entities["anymal_d"] = Articulation(cfg=anymal_d_cfg)

    go2_cfg = UNITREE_GO2_CFG
    go2_cfg.prim_path = "/World/scene/go2"
    go2_cfg.spawn = None
    scene_entities["go2"] = Articulation(cfg=go2_cfg)

    a1_cfg = UNITREE_A1_CFG
    a1_cfg.prim_path = "/World/scene/a1"
    a1_cfg.spawn = None
    scene_entities["a1"] = Articulation(cfg=a1_cfg)

    franka_cfg = FRANKA_PANDA_CFG
    franka_cfg.prim_path = "/World/scene/franka_panda"
    franka_cfg.spawn = None
    scene_entities["franka"] = Articulation(cfg=franka_cfg)

    ur10_cfg = UR10_CFG
    ur10_cfg.prim_path = "/World/scene/ur10"
    ur10_cfg.spawn = None
    scene_entities["ur10"] = Articulation(cfg=ur10_cfg)

    gen3n7_cfg = KINOVA_GEN3_N7_CFG
    gen3n7_cfg.prim_path = "/World/scene/gen3n7"
    gen3n7_cfg.spawn = None
    scene_entities["gen3n7"] = Articulation(cfg=gen3n7_cfg)

    sawyer_cfg = SAWYER_CFG
    sawyer_cfg.prim_path = "/World/scene/sawyer"
    sawyer_cfg.spawn = None
    scene_entities["sawyer"] = Articulation(cfg=sawyer_cfg)

    h1_cfg = H1_CFG
    h1_cfg.prim_path = "/World/scene/h1"
    h1_cfg.spawn = None
    scene_entities["h1"] = Articulation(cfg=h1_cfg)

    # return the scene information
    return scene_entities


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    for robot in entities.values():
        robot.data.default_root_state = robot.data.root_state_w.clone()

    elbow_indices, _ = entities["h1"].find_joints(".*_elbow")
    default_joint_pos = entities["h1"].data.default_joint_pos.clone()
    default_joint_pos[0, elbow_indices] = -0.2
    entities["h1"].data.default_joint_pos = default_joint_pos

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for robot in entities.values():
                # root state
                root_state = robot.data.default_root_state.clone()
                robot.write_root_state_to_sim(root_state)
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
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()

    _ = IsaacSimPublisher(
        host="127.0.0.1",
        stage=sim.stage,
        ignored_prim_paths=[
            "/World/scene/GroundPlane",
            "/World/scene/Cube_04",
            "/World/scene/Cube_05",
            "/World/scene/Cube_06",
        ],
        texture_cache_dir=f"{Path(__file__).parent.as_posix()}/.textures",
    )

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
