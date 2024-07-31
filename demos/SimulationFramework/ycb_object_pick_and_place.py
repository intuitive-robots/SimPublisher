import argparse
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
from simpub.sim.sf_publisher import SFPublisher

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    host = args.host

    ycb_base_folder = "/home/xinkai/project/SF-ObjectDataset/YCB"
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
    scene = sim_factory.create_scene(object_list=object_list, dt=0.0005)
    robot = sim_factory.create_robot(scene, dt=0.0005)
    scene.start()

    assert host is not None, "Please specify the host"
    publisher = SFPublisher(
        scene, host, no_tracked_objects=["table_plane", "table0"]
    )
    publisher.start()

    robot.set_desired_gripper_width(0.4)

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
