from pathlib import Path
from typing import Optional, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

from scipy.spatial.transform import Rotation as R

# Function to apply Z-axis rotation to a quaternion
def apply_z_rotation(quat, z_angle=np.pi / 2):
    """
    Apply a rotation around the Z-axis to a given quaternion.

    Args:
        quat: The original quaternion (x, y, z, w).
        z_angle: The rotation angle around the Z-axis in radians.

    Returns:
        A new quaternion after applying the Z-axis rotation.
    """
    # Convert the input quaternion to a rotation object
    rotation = R.from_quat(quat)

    # Create a rotation around the Z-axis
    z_rotation = R.from_euler("z", z_angle)

    # Combine the rotations
    new_rotation = (
        rotation * z_rotation
    )  # Order matters: z_rotation is applied first

    # Convert back to quaternion
    return new_rotation.as_quat()


_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        subtree_ids: List of subtree ids. A subtree is defined as the kinematic tree
            starting at the body and including all its descendants. Gravity
            compensation forces will be applied to all bodies in the subtree.
        qfrc_applied: Optional array to store the computed forces. If not provided,
            the applied forces in `data` are used.
    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    publisher = MujocoPublisher(model, data, host="192.168.0.134")
    mq3 = MetaQuest3("IRLMQ3-1")

    # Bodies for which to apply gravity compensation.
    left_subtree_id = model.body("left/base_link").id
    right_subtree_id = model.body("right/base_link").id

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names: list[str] = []
    velocity_limits: dict[str, float] = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    # Enable collision avoidance between the following geoms.
    l_wrist_geoms = mink.get_subtree_geom_ids(
        model, model.body("left/wrist_link").id
    )
    r_wrist_geoms = mink.get_subtree_geom_ids(
        model, model.body("right/wrist_link").id
    )
    l_geoms = mink.get_subtree_geom_ids(
        model, model.body("left/upper_arm_link").id
    )
    r_geoms = mink.get_subtree_geom_ids(
        model, model.body("right/upper_arm_link").id
    )
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_avoidance_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5

    left_gripper_actuator = model.actuator("left/gripper").id
    right_gripper_actuator = model.actuator("right/gripper").id

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(
            model, data, "left/target", "left/gripper", "site"
        )
        mink.move_mocap_to_frame(
            model, data, "right/target", "right/gripper", "site"
        )

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets.
            l_ee_task.set_target(
                mink.SE3.from_mocap_name(model, data, "left/target")
            )
            r_ee_task.set_target(
                mink.SE3.from_mocap_name(model, data, "right/target")
            )

            # Update posture task target.
            input_data = mq3.get_input_data()
            if input_data is not None:
                left_hand = input_data["left"]
                right_hand = input_data["right"]
                if left_hand["hand_trigger"]:
                    pos = np.array(input_data["left"]["pos"])
                    pos[0] = pos[0] + 0.1
                    data.mocap_pos[model.body("left/target").mocapid[0]] = pos
                    rot = input_data["left"]["rot"]
                    rot = apply_z_rotation(rot, z_angle=-np.pi / 2)
                    data.mocap_quat[
                        model.body("left/target").mocapid[0]
                    ] = np.array([rot[3], rot[0], rot[1], rot[2]])
                    if left_hand["index_trigger"]:
                        data.ctrl[left_gripper_actuator] = 0.002
                    else:
                        data.ctrl[left_gripper_actuator] = 0.037
                if right_hand["hand_trigger"]:
                    pos = np.array(input_data["right"]["pos"])
                    pos[0] = pos[0] - 0.1
                    data.mocap_pos[model.body("right/target").mocapid[0]] = pos
                    rot = input_data["right"]["rot"]
                    rot = apply_z_rotation(rot, z_angle=np.pi / 2)
                    data.mocap_quat[
                        model.body("right/target").mocapid[0]
                    ] = np.array([rot[3], rot[0], rot[1], rot[2]])
                    if right_hand["index_trigger"]:
                        data.ctrl[right_gripper_actuator] = 0.002
                    else:
                        data.ctrl[right_gripper_actuator] = 0.037

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = l_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if (
                    l_pos_achieved
                    and l_ori_achieved
                    and r_pos_achieved
                    and r_ori_achieved
                ):
                    break

            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            compensate_gravity(
                model, data, [left_subtree_id, right_subtree_id]
            )
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            # rate.sleep()
