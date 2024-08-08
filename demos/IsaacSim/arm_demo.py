# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import io
from hashlib import md5
import trimesh
import random
import math

from simpub.core.simpub_server import SimPublisher
from simpub.simdata import (
    SimScene,
    SimObject,
    SimVisual,
    VisualType,
    SimTransform,
    SimMesh,
)

from omni.isaac.lab.app import AppLauncher

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

"""Rest everything follows."""

import numpy as np
import torch

import omni.usd
from pxr import Usd, UsdGeom, UsdUtils, UsdPhysics, Gf
import numpy as np
import omni
import carb
from omni.physx.scripts import utils
from omni.physx import get_physx_interface

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

#
import gymnasium as gym
import torch

import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
#

print("=" * 20)
print(ISAACLAB_NUCLEUS_DIR)

##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # currently this function is unimportant, since we only test with a single origin/env.

    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(
        torch.arange(num_rows), torch.arange(num_cols), indexing="xy"
    )
    env_origins[:, 0] = (
        spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    )
    env_origins[:, 1] = (
        spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    )
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # this function will build the scene by adding primitives to it.

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=2.0)

    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    prim_utils.create_prim("/World/Origin1/Tables", "Xform")

    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    cfg.func("/World/Origin1/Tables/Table", cfg, translation=(0.55, 0.0, 1.05))
    cfg.func("/World/Origin1/Tables/Table_1", cfg, translation=(0.55, 3.0, 1.05))

    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_panda = Articulation(cfg=franka_arm_cfg)

    # -- cube
    cfg_cube = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )
    cfg_cube.func("/World/Origin1/Cube1", cfg_cube, translation=(0.2, 0.0, 3.0))

    # return the scene information
    scene_entities = {
        "franka_panda": franka_panda,
    }
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
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_state_to_sim(root_state)
                # set joint positions
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = (
                robot.data.default_joint_pos
                + torch.randn_like(robot.data.joint_pos) * 0.1
            )
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0],
                robot.data.soft_joint_pos_limits[..., 1],
            )
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


class IsaacSimPublisher(SimPublisher):
    def __init__(self, host: str, stage: Usd.Stage) -> None:
        self.tracked_prims: list[dict] = []
        sim_scene = self.parse_scene(stage)
        super().__init__(sim_scene, host)

    def parse_scene(self, stage: Usd.Stage) -> SimScene:
        print("=" * 50)
        print("parsing stage:", stage)

        self.use_usdrt_stage(stage)

        scene = SimScene()
        self.sim_scene = scene

        scene.root = SimObject(name="root", trans=SimTransform())

        obj1 = SimObject(name="object_1", trans=SimTransform(pos=[10, 0, 0]))
        obj1.visuals.append(
            SimVisual(
                type=VisualType.CUBE,
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(),
            )
        )
        scene.root.children.append(obj1)

        bin_buffer = io.BytesIO()

        mesh = trimesh.creation.box(extents=[1, 2, 3])

        indices = mesh.faces.astype(np.int32)
        bin_buffer = io.BytesIO()

        # Vertices
        verts = mesh.vertices.astype(np.float32)
        verts[:, 2] = -verts[:, 2]
        verts = verts.flatten()
        vertices_layout = bin_buffer.tell(), verts.shape[0]
        bin_buffer.write(verts)

        # Normals
        norms = mesh.vertex_normals.astype(np.float32)
        norms[:, 2] = -norms[:, 2]
        norms = norms.flatten()
        normal_layout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms)

        # Indices
        indices = mesh.faces.astype(np.int32)
        indices = indices[:, [2, 1, 0]]
        indices = indices.flatten()
        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)

        # # Texture coords
        # uv_layout = (0, 0)
        # if hasattr(mesh.visual, "uv"):
        #     uvs = mesh.visual.uv.astype(np.float32)
        #     uvs[:, 1] = 1 - uvs[:, 1]
        #     uvs = uvs.flatten()
        #     uv_layout = bin_buffer.tell(), uvs.shape[0]

        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()

        mesh = SimMesh(
            id="mesh_1",
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            dataHash=hash,
            normalsLayout=normal_layout,
            uvLayout=(0, 0),
        )

        self.sim_scene.meshes.append(mesh)
        self.sim_scene.raw_data[mesh.dataHash] = bin_data

        obj2 = SimObject(name="object_2")
        obj2.visuals.append(
            SimVisual(
                type=VisualType.MESH,
                mesh="mesh_1",
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(pos=[-10, 0, 0]),
            )
        )
        scene.root.children.append(obj2)

        # root_path = "/World/Origin1"
        root_path = "/World"
        obj2 = self.parse_prim_tree(root=stage.GetPrimAtPath(root_path))
        assert obj2 is not None
        scene.root.children.append(obj2)

        return scene

    def use_usdrt_stage(self, stage: Usd.Stage):
        import omni
        import omni.usd

        from pxr import Usd, UsdUtils
        from usdrt import Usd as RtUsd
        from usdrt import UsdGeom as RtGeom
        from usdrt import Rt

        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        stage_id = stage_id.ToLongInt()
        print("usdrt stage id:", stage_id)

        rtstage = RtUsd.Stage.Attach(stage_id)
        print("usdrt stage:", rtstage)

        self.rt_stage = rtstage

    def parse_prim_tree(
        self,
        root: Usd.Prim,
        indent=0,
        parent_path=None,
    ) -> SimObject | None:
        if root.GetTypeName() not in {"Xform", "Mesh", "Scope"}:  # Cube
            # not good...
            # perhaps traverse twice and preserve only prims with meshes as children
            if root.GetName() != "World":
                return

        purpose_attr = root.GetAttribute("purpose")
        if purpose_attr and purpose_attr.Get() in {"proxy", "guide"}:
            return

        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

        trans_mat = omni.usd.get_local_transform_matrix(root, timecode)

        # row_y, row_z = trans_mat.GetRow(1), trans_mat.GetRow(2)
        # trans_mat.SetRow(1, row_z)
        # trans_mat.SetRow(2, row_y)
        # col_y, col_z = trans_mat.GetColumn(1), trans_mat.GetColumn(2)
        # trans_mat.SetColumn(1, col_z)
        # trans_mat.SetColumn(2, col_y)

        # print(trans_mat.IsLeftHanded(), trans_mat.IsRightHanded())
        # print(trans_mat)

        x_scale = Gf.Vec3d(
            trans_mat[0][0], trans_mat[0][1], trans_mat[0][2]
        ).GetLength()
        y_scale = Gf.Vec3d(
            trans_mat[1][0], trans_mat[1][1], trans_mat[1][2]
        ).GetLength()
        z_scale = Gf.Vec3d(
            trans_mat[2][0], trans_mat[2][1], trans_mat[2][2]
        ).GetLength()
        scale = [x_scale, z_scale, y_scale]
        # print("\t" * indent, x_scale, y_scale, z_scale)

        translate = trans_mat.ExtractTranslation()
        translate = [-translate[1], translate[2], translate[0]]

        rot = trans_mat.ExtractRotationQuat()
        imag = rot.GetImaginary()
        rot = [imag[1], -imag[2], -imag[0], rot.GetReal()]

        if parent_path is None:
            prim_path = str(root.GetPrimPath())
        else:
            prim_path = f"{parent_path}/{root.GetName()}"

        sim_object = SimObject(
            name=prim_path.replace("/", "_"),
            trans=SimTransform(pos=translate, rot=rot, scale=scale),
        )

        print(
            "\t" * indent
            + f"{prim_path}: {root.GetTypeName()} {root.GetAttribute('purpose').Get()}"
        )

        # maybe time_code is necessary
        # trans_mat = omni.usd.get_local_transform_matrix(root)
        # print("\t" * indent + f"{trans_mat}")

        # attr: Usd.Property
        # for attr in root.GetProperties():
        #     print("\t" * indent + f"{attr.GetName()}")

        if root.GetTypeName() == "Mesh":
            mesh_prim = UsdGeom.Mesh(root)
            assert mesh_prim is not None

            points = np.asarray(mesh_prim.GetPointsAttr().Get()).astype(np.float32)
            # points[:, [1, 2]] = points[:, [2, 1]]

            normals = np.asarray(mesh_prim.GetNormalsAttr().Get()).astype(np.float32)
            # normals[:, [1, 2]] = normals[:, [2, 1]]

            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).astype(
                np.int32
            )

            face_vertex_counts = np.asarray(
                mesh_prim.GetFaceVertexCountsAttr().Get()
            ).astype(np.int32)

            # either only triangular faces or only quad faces
            assert len(set(face_vertex_counts)) == 1
            num_vert_per_face = face_vertex_counts[0]

            mesh_obj = trimesh.Trimesh(
                vertices=points, faces=indices.reshape(-1, num_vert_per_face)
            )
            mesh_obj = mesh_obj.apply_transform(
                trimesh.transformations.euler_matrix(-math.pi / 2.0, math.pi / 2.0, 0)
            )
            mesh_obj.fix_normals()

            points = points.flatten()
            normals = normals.flatten()
            indices = indices.flatten()

            print(
                "\t" * indent
                + f"vertices size: {points.shape[0] // 3} {points.shape[0] % 3}"
            )
            print(
                "\t" * indent
                + f"normals size: {normals.shape[0] // 3} {normals.shape[0] % 3}"
            )
            print(
                "\t" * indent
                + f"triangles: {indices.shape[0] // 3} {indices.shape[0] % 3}"
            )

            assert normals.shape[0] // 3 == indices.shape[0]
            print(
                "\t" * indent
                + f"normal per index: {normals.shape[0] // 3} {indices.shape[0]}"
            )
            # print(
            #     "\t" * indent + f"@vert {points} {points.dtype} {points.shape}"
            # )
            # print(
            #     "\t" * indent + f"@indi {indices} {indices.dtype} {indices.shape}"
            # )

            # bin_buffer = io.BytesIO()

            # # Vertices
            # vertices_layout = bin_buffer.tell(), points.shape[0]
            # bin_buffer.write(points)
            # print("\t" * indent + f"vertices layout: {vertices_layout}")

            # normals_layout = bin_buffer.tell(), normals.shape[0]
            # bin_buffer.write(normals)
            # print("\t" * indent + f"normals layout: {normals_layout}")

            # # Indices
            # indices_layout = bin_buffer.tell(), indices.shape[0]
            # bin_buffer.write(indices)

            # bin_data = bin_buffer.getvalue()
            # hash = md5(bin_data).hexdigest()

            # mesh_id = "@mesh-" + str(random.randint(int(1e9), int(1e10 - 1)))
            # mesh = SimMesh(
            #     id=mesh_id,
            #     indicesLayout=indices_layout,
            #     verticesLayout=vertices_layout,
            #     dataHash=hash,
            #     normalsLayout=(0, 0),
            #     uvLayout=(0, 0),
            # )

            indices = mesh_obj.faces.astype(np.int32)
            bin_buffer = io.BytesIO()
            # Vertices
            verts = mesh_obj.vertices.astype(np.float32)
            verts[:, 2] = -verts[:, 2]
            verts = verts.flatten()
            vertices_layout = bin_buffer.tell(), verts.shape[0]
            bin_buffer.write(verts)
            # Normals
            norms = mesh_obj.vertex_normals.astype(np.float32)
            norms[:, 2] = -norms[:, 2]
            norms = norms.flatten()
            normal_layout = bin_buffer.tell(), norms.shape[0]
            bin_buffer.write(norms)
            # Indices
            indices = mesh_obj.faces.astype(np.int32)
            indices = indices[:, [2, 1, 0]]
            indices = indices.flatten()
            indices_layout = bin_buffer.tell(), indices.shape[0]
            bin_buffer.write(indices)
            # Texture coords
            uv_layout = (0, 0)
            if hasattr(mesh_obj.visual, "uv"):
                uvs = mesh_obj.visual.uv.astype(np.float32)
                uvs[:, 1] = 1 - uvs[:, 1]
                uvs = uvs.flatten()
                uv_layout = bin_buffer.tell(), uvs.shape[0]

            bin_data = bin_buffer.getvalue()
            hash = md5(bin_data).hexdigest()

            mesh_id = "@mesh-" + str(random.randint(int(1e9), int(1e10 - 1)))
            mesh = SimMesh(
                id=mesh_id,
                indicesLayout=indices_layout,
                verticesLayout=vertices_layout,
                normalsLayout=normal_layout,
                uvLayout=uv_layout,
                dataHash=hash,
            )

            self.sim_scene.meshes.append(mesh)
            self.sim_scene.raw_data[mesh.dataHash] = bin_data

            #!
            #! do not create new mesh when multiple primitives point to the same prototype
            #!

            sim_mesh = SimVisual(
                type=VisualType.MESH,
                mesh=mesh_id,
                color=[1.0, 1.0, 1.0, 1.0],
                trans=SimTransform(),
            )
            sim_object.visuals.append(sim_mesh)

        # track prims with rigid objects attached
        if (attr := root.GetAttribute("physics:rigidBodyEnabled")) and attr.Get():
            print(f"tracking {sim_object.name}")
            self.tracked_prims.append(
                {"name": sim_object.name, "prim": root, prim_path: ""}
            )

        child: Usd.Prim

        if root.IsInstance():
            proto = root.GetPrototype()
            print("\t" * indent + f"@prototype: {proto.GetPrimPath()}")

            for child in proto.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child, indent=indent + 1, parent_path=prim_path
                ):
                    sim_object.children.append(obj)

        else:
            for child in root.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child, indent=indent + 1, parent_path=prim_path
                ):
                    sim_object.children.append(obj)

        return sim_object

    def get_update(self) -> dict[str, list[float]]:
        def print_state():
            import omni
            import omni.usd

            from pxr import Usd, UsdUtils
            from usdrt import Usd as RtUsd
            from usdrt import UsdGeom as RtGeom
            from usdrt import Rt

            prim = self.rt_stage.GetPrimAtPath("/World/Origin1/Robot/panda_hand")
            print(prim)
            print(prim.GetTypeName())

            prim = Rt.Xformable(prim)
            print(prim.GetWorldPositionAttr().Get())

        print_state()

        state = {}

        # for name, trans in self.tracked_obj_trans.items():
        #     pos, rot = trans
        #     state[name] = [-pos[1], pos[2], pos[0], rot[2], -rot[3], -rot[1], rot[0]]

        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

        # print()
        # print(timecode)
        for tracked_prim in self.tracked_prims:
            prim_name = tracked_prim["name"]
            # prim_path = tracked_prim["path"]
            prim = tracked_prim["prim"]

            # cur_trans = get_physx_interface().get_rigidbody_transformation(prim_path)
            # print(cur_trans)

            trans_mat = omni.usd.get_world_transform_matrix(prim, timecode)
            # print(f"{prim_name}: {trans_mat}")

            translate = trans_mat.ExtractTranslation()
            translate = [-translate[1], translate[2], translate[0]]

            rot = trans_mat.ExtractRotationQuat()
            imag = rot.GetImaginary()
            rot = [imag[1], -imag[2], -imag[0], rot.GetReal()]

            state[prim_name] = [
                translate[0],
                translate[1],
                translate[2],
                rot[0],
                rot[1],
                rot[2],
                rot[3],
            ]

        return state


def parse_stage(stage: Usd.Stage):
    # publisher = IsaacSimPublisher(host="192.168.0.134", stage=stage)
    publisher = IsaacSimPublisher(host="127.0.0.1", stage=stage)


def run_custom_scene():
    """Main function."""
    # sim_utils.SimulationContext is a singleton class
    # if SimulationContext.instance() is None:
    #     self.sim: SimulationContext = SimulationContext(self.cfg.sim)
    # or: print(env.sim)

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    # sim_cfg.use_fabric = False
    # sim_cfg.device = "cpu"
    # sim_cfg.use_gpu_pipeline = False
    # sim_cfg.physx.use_gpu = False
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    parse_stage(sim.stage)

    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


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


def run_sample_env_1():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    task = "Isaac-Lift-Cube-Franka-IK-Rel-v0"

    # parse configuration
    env_cfg = parse_env_cfg(
        task_name=task,
        use_gpu=False,
        num_envs=1,
        use_fabric=False,
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
        parse_stage(env.sim.stage)

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
    # run simulation on sample scenes

    run_custom_scene()
    # run_sample_env_1()

    # close sim app
    simulation_app.close()


# stage: Usd.Stage = omni.usd.get_context().get_stage()
# print(f"stage: {stage}\n")


# # def iterate_prim_children(root: Usd.Prim):
# #     print(f"{root.GetName()}")
# #     obj: Usd.Prim
# #     for obj in root.GetChildren():
# #         print("")
# #         print("=" * 50)
# #         print(f"obj: {obj}")
# #         print(f"type: {obj.GetTypeName()}")
# #         print(f"is instance: {obj.IsInstance()}")
# #         print(f"is instance proxy: {obj.IsInstanceProxy()}")
# #         print(f"is instancable: {obj.IsInstanceable()}")
# #         if obj.IsInstance():
# #             print(obj.GetPrototype().IsInPrototype())
# #             print(obj.GetPrototype().GetChildrenNames())
# #             print(obj.GetPrototype().GetTypeName())
# #             for child in obj.GetPrototype().GetChildren():
# #                 print(f"{child.GetName()} {child.GetTypeName()}")
# #         else:
# #             print(obj.GetChildrenNames())
# #             for i, name in enumerate(obj.GetPropertyNames()):
# #                 print(name)

# #         if obj.IsInstance():
# #             iterate_prim_children(obj.GetPrototype())
# #         else:
# #             iterate_prim_children(obj)


# # iterate_prim_children(stage.GetPrimAtPath("/World/Origin1"))

# # timeline = omni.timeline.get_timeline_interface()
# # timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

# # prim = stage.GetPrimAtPath("/World/Origin1/Table/Visuals")

# # print(omni.usd.get_world_transform_matrix(prim, timecode))
# # print(prim)
# # print(prim.IsInstance())
# # print(prim.GetTypeName())
# # print(prim.GetPrototype())
# # proto = prim.GetPrototype()
# # print(proto.GetChildrenNames())
# # geom = proto.GetChild("TableGeom")
# # print(geom)
# # print(geom.GetTypeName())
# # print(geom.GetChildrenNames())
# # print(geom.GetChild("subset").GetChildrenNames())
# # mesh = UsdGeom.Mesh(geom)
# # print(mesh)
# # points = np.asarray(mesh.GetPointsAttr().Get())
# # indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
# # print(f"{points} {points.dtype} {points.shape}")
# # print(f"{indices} {indices.dtype} {indices.shape}")
