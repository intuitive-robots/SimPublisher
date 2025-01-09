from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
import os
import argparse
import zmq
import threading
import json
import traceback

from simpub.sim.sf_publisher import SFPublisher
from alr_sim.controllers.Controller import JointPDController


class RealRobotVTController(JointPDController):
    def __init__(self, real_robot_ip):
        super().__init__()
        self.sub_socket = zmq.Context().socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{real_robot_ip}:5555")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.data = None
        self.sub_task = threading.Thread(target=self.subscribe_task)
        self.sub_task.start()

    def subscribe_task(self):
        try:
            while True:
                msg = self.sub_socket.recv_string()
                self.data = json.loads(msg)
        except Exception:
            traceback.print_exc()

    def getControl(self, robot):
        if self.data is None:
            return super().getControl(robot)
        self.setSetPoint(
            desired_pos=self.data['q'], desired_vel=self.data['dq']
        )
        if self.data['gripper_width'][0] < 0.9 * self.data['gripper_width'][1]:
            robot.set_gripper_cmd_type = 2  # Move
        else:
            robot.set_gripper_cmd_type = 1  # Grasp
        robot.set_gripper_width = self.goal_gripper_width_fct()
        return super().getControl(robot)


if __name__ == "__main__":

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    ycb_base_folder = os.path.join(args.folder, "SF-ObjectDataset/YCB")

    hammer = YCBMujocoObject(
        ycb_base_folder=ycb_base_folder,
        object_id="048_hammer",
        object_name="hammer",
        pos=[0.4, 0.0, 0.1],
        quat=[0, 0, 0, 1],
        static=False,
        alpha=1.0,
        visual_only=False,
    )
    object_list = [hammer]

    # Setup the scene
    sim_factory = SimRepository.get_factory("mj_beta")

    # Setting the dt to 0.0005 to reduce jittering of the gripper due to more difficult Physics Simulation
    scene = sim_factory.create_scene(object_list=object_list, dt=0.001)
    robot1 = sim_factory.create_robot(
        scene,
        dt=0.001,
        base_position=[0.0, 0.5, 0.0]
    )
    robot2 = sim_factory.create_robot(
        scene,
        dt=0.001,
        base_position=[0.0, -0.5, 0.0]
    )
    controller1 = RealRobotVTController("")
    controller2 = RealRobotVTController("")
    scene.start()
    controller1.executeController(robot1, maxDuration=1000, block=False)
    controller2.executeController(robot2, maxDuration=1000, block=False)
    publisher = SFPublisher(scene, args.host)

    try:
        while True:
            scene.next_step()
    except KeyboardInterrupt:
        publisher.shutdown()
        print("Simulation finished")
