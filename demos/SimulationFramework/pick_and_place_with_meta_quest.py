import argparse
import numpy as np

from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.controllers.IKControllers import CartPosQuatImpedenceController
from simpub.sim.sf_publisher import SFPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


class MetaQuest3Controller(CartPosQuatImpedenceController):

    def __init__(self, device):
        super().__init__()
        self.device: MetaQuest3 = device

    def getControl(self, robot):
        input_data = self.device.get_input_data()
        if input_data is not None:
            desired_pos = input_data.right_pos
            desired_quat = input_data.right_rot
            desired_pos_local = robot._localize_cart_pos(desired_pos)
            desired_quat_local = robot._localize_cart_quat(desired_quat)
            desired_quat_local = [0, 1, 0, 0]
            if input_data.right_index_trigger:
                robot.close_fingers(duration=0.0)
            else:
                robot.open_fingers()
            self.setSetPoint(np.hstack((desired_pos_local, desired_quat_local)))
            
        return super().getControl(robot)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    box1 = Box(
        name="box1",
        init_pos=[0.5, -0.2, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[0.1, 0.25, 0.3, 1],
    )
    box2 = Box(
        name="box2",
        init_pos=[0.6, -0.1, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[0.2, 0.3, 0.7, 1],
    )
    box3 = Box(
        name="box3",
        init_pos=[0.4, -0.1, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )
    box4 = Box(
        name="box4",
        init_pos=[0.6, -0.0, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )
    box5 = Box(
        name="box5",
        init_pos=[0.6, 0.1, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 1, 1, 1],
    )
    box6 = Box(
        name="box6",
        init_pos=[0.6, 0.2, 0.5],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1],
    )

    table = Box(
        name="table0",
        init_pos=[0.5, 0.0, 0.2],
        init_quat=[0, 1, 0, 0],
        size=[0.25, 0.35, 0.2],
        static=True,
    )

    object_list = [box1, box2, box3, box4, box5, box6, table]

    # Setup the scene
    sim_factory = SimRepository.get_factory("mj_beta")

    scene = sim_factory.create_scene(object_list=object_list)
    robot = sim_factory.create_robot(scene)

    scene.start()

    publisher = SFPublisher(
        scene, args.host, no_tracked_objects=["table_plane", "table0"]
    )
    meta_quest3 = MetaQuest3(publisher, "192.168.0.102")
    # meta_quest3 = MetaQuest3(publisher, "192.168.0.143")
    robot_controller = MetaQuest3Controller(meta_quest3)
    robot_controller.executeController(robot, maxDuration=1000, block=False)
    publisher.start()

    while True:
        scene.next_step()
