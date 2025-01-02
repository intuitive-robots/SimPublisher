from alr_sim.core.logger import RobotPlotFlags
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
import os
import argparse


from simpub.sim.sf_publisher import SFPublisher

if __name__ == "__main__":

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--object_id", type=str, default="013_apple")
    args = parser.parse_args()

    ycb_base_folder = os.path.join(args.folder, "SF-ObjectDataset/YCB")
    clamp = YCBMujocoObject(
        ycb_base_folder=ycb_base_folder,
        object_id="051_large_clamp",
        object_name="clamp",
        pos=[0.4, 0, 0.1],
        quat=[0, 0, 0, 1],
        static=False,
        alpha=1.0,
        visual_only=False,
    )

    lemon = YCBMujocoObject(
        ycb_base_folder=ycb_base_folder,
        object_id="014_lemon",
        object_name="lemon",
        pos=[0.4, 0.2, 0.1],
        quat=[0, 0, 0, 1],
        static=False,
        alpha=1.0,
        visual_only=False,
    )

    mug = YCBMujocoObject(
        ycb_base_folder=ycb_base_folder,
        object_id="025_mug",
        object_name="mug",
        pos=[0.2, 0.1, 0.1],
        quat=[0, 0, 0, 1],
        static=False,
        alpha=1.0,
        visual_only=False,
    )

    hammer = YCBMujocoObject(
        ycb_base_folder=ycb_base_folder,
        object_id="048_hammer",
        object_name="hammer",
        pos=[0.3, -0.2, 0.1],
        quat=[0, 0, 0, 1],
        static=False,
        alpha=1.0,
        visual_only=False,
    )

    object_list = [clamp, lemon, mug, hammer]

    # Setup the scene
    sim_factory = SimRepository.get_factory("mj_beta")

    # Setting the dt to 0.0005 to reduce jittering of the gripper due to more difficult Physics Simulation
    scene = sim_factory.create_scene(object_list=object_list, dt=0.0005)
    robot = sim_factory.create_robot(scene, dt=0.0005)
    scene.start()

    publisher = SFPublisher(scene, args.host)

    robot.set_desired_gripper_width(0.4)  # we set the gripper to clos at the beginning

    # execute the pick and place movements
    robot.gotoCartPositionAndQuat(
        [0.4, 0, 0.1], [0, 0.7071068, -0.7071068, 0], duration=8
    )
    robot.gotoCartPositionAndQuat(
        [0.4, 0, -0.01], [0, 0.7071068, -0.7071068, 0], duration=2
    )
    robot.close_fingers()
    robot.gotoCartPositionAndQuat(
        [0.4, 0, 0.1], [0, 0.7071068, -0.7071068, 0], duration=8
    )
    robot.gotoCartPositionAndQuat(
        [0.8, 0, 0.1], [0, 0.7071068, -0.7071068, 0], duration=4
    )

    robot.open_fingers()

    while True:
        scene.next_step()
