from alr_sim.core.logger import RobotPlotFlags
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
from simpub.simulation_framework.sf_publisher import SFPublisher
import os

if __name__ == "__main__":
    ######################################################################################
    # THIS DEMO REQUIRES YOU TO DOWNLOAD THE SF-ObjectDataset somewhere on your computer #
    ######################################################################################
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

    object_list = [clamp]

    # Setup the scene
    sim_factory = SimRepository.get_factory("mj_beta")

    # Setting the dt to 0.0005 to reduce jittering of the gripper due to more difficult Physics Simulation
    scene = sim_factory.create_scene(object_list=object_list, dt=0.0005)
    robot = sim_factory.create_robot(scene, dt=0.0005)
    scene.start()

    publisher = SFPublisher(scene)
    publisher.start()

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

    robot.wait(10)

    scene.stop_logging()
