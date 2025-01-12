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
from omni.isaac.lab.assets import Articulation
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


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    env_quats = torch.zeros(num_origins, 4)
    # compute origins
    x = 0
    for i in range(num_origins):
        env_origins[i, 0] = x
        env_quats[i, :] = torch.tensor(euler_angles_to_quat([0, 0, -0.5 * math.pi]))
        x += spacing
    # return the origins
    return env_origins.tolist(), env_quats.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origin_x = 0
    origin_y = 0
    origins = []
    orientation = euler_angles_to_quat([0, 0, 0.5 * math.pi])
    i_origin = -1
    scene_entities = {}

    #
    # quadrupeds
    #

    # Origin 1 with Anymal B
    i_origin += 1
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    anymal_b_cfg = ANYMAL_B_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    anymal_b_cfg.init_state.rot = orientation
    anymal_b = Articulation(cfg=anymal_b_cfg)
    scene_entities["anymal_b"] = anymal_b

    # Origin 2 with Anymal C
    i_origin += 1
    origin_x += 1.2
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    anymal_c_cfg = ANYMAL_C_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    anymal_c_cfg.init_state.rot = orientation
    anymal_c = Articulation(cfg=anymal_c_cfg)
    scene_entities["anymal_c"] = anymal_c

    # Origin 3 with Anymal D
    i_origin += 1
    origin_x += 1.2
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    anymal_d_cfg = ANYMAL_D_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    anymal_d_cfg.init_state.rot = orientation
    anymal_d = Articulation(cfg=anymal_d_cfg)
    scene_entities["anymal_d"] = anymal_d

    # Origin 4 with Unitree A1
    i_origin += 1
    origin_x += 0.9
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    unitree_a1_cfg = UNITREE_A1_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    unitree_a1_cfg.init_state.rot = orientation
    unitree_a1 = Articulation(cfg=unitree_a1_cfg)
    scene_entities["unitree_a1"] = unitree_a1

    # Origin 5 with Unitree Go2
    i_origin += 1
    origin_x += 0.9
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    unitree_go2_cfg = UNITREE_GO2_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    unitree_go2_cfg.init_state.rot = orientation
    unitree_go2 = Articulation(cfg=unitree_go2_cfg)
    scene_entities["unitree_go2"] = unitree_go2

    # Origin 6 with Boston Dynamics Spot
    i_origin += 1
    origin_x += 0.9
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Robot
    spot_cfg = SPOT_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    spot_cfg.init_state.rot = orientation
    spot = Articulation(cfg=spot_cfg)
    scene_entities["spot"] = spot

    #
    # robot arms
    #

    # Franka Panda
    i_origin += 1
    origin_y -= 2.5
    origin_x = 0
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func(f"/World/Origin{i_origin}/Table", cfg, translation=(0.55, 0.0, 1.05))
    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_arm_cfg.init_state.rot = orientation
    franka_panda = Articulation(cfg=franka_arm_cfg)
    scene_entities["franka_panda"] = franka_panda

    # UR10
    i_origin += 1
    origin_x += 1.5
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func(f"/World/Origin{i_origin}/Table", cfg, translation=(0.0, 0.0, 1.03))
    # -- Robot
    ur10_cfg = UR10_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    ur10_cfg.init_state.rot = orientation
    ur10 = Articulation(cfg=ur10_cfg)
    scene_entities["ur10"] = ur10

    # Sawyer
    i_origin += 1
    origin_x += 0.7
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    # -- Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func(f"/World/Origin{i_origin}/Table", cfg, translation=(0.55, 0.0, 1.05))
    # -- Robot
    kinova_arm_cfg = KINOVA_GEN3_N7_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    kinova_arm_cfg.init_state.rot = orientation
    kinova_gen3n7 = Articulation(cfg=kinova_arm_cfg)
    scene_entities["kinova_gen3n7"] = kinova_gen3n7

    # Kinova Gen3 (7-Dof) arm
    i_origin += 1
    origin_x += 1.5
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(
        f"/World/Origin{i_origin}",
        "Xform",
        translation=origins[-1],
        orientation=orientation,
    )
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func(f"/World/Origin{i_origin}/Table", cfg, translation=(0.0, 0.0, 1.03))
    # -- Robot
    sawyer_arm_cfg = SAWYER_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    sawyer_arm_cfg.init_state.pos = (0.0, 0.0, 1.03)
    sawyer_arm_cfg.init_state.rot = orientation
    sawyer = Articulation(cfg=sawyer_arm_cfg)
    scene_entities["sawyer"] = sawyer

    #
    # humanoid robots
    #

    # H1
    i_origin += 1
    origin_x += 0.7
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    h1_cfg = H1_MINIMAL_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    h1_cfg.init_state.rot = orientation
    h1 = Articulation(cfg=h1_cfg)
    scene_entities["h1"] = h1

    # G1
    i_origin += 1
    origin_x += 0.7
    origins.append([origin_x, origin_y, 0])
    prim_utils.create_prim(f"/World/Origin{i_origin}", "Xform", translation=origins[-1])
    g1_cfg = G1_MINIMAL_CFG.replace(prim_path=f"/World/Origin{i_origin}/Robot")
    g1_cfg.init_state.rot = orientation
    g1 = Articulation(cfg=g1_cfg)
    scene_entities["g1"] = g1

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
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
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
