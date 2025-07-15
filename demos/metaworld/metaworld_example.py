import argparse
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import metaworld
from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


class MQ3CartController:
    def __init__(self, meta_quest3: MetaQuest3):
        self.meta_quest3 = meta_quest3
        self.last_state = None

    def get_action(self):
        input_data = self.meta_quest3.get_input_data()
        action = np.zeros(4)
        if input_data is None:
            return action
        hand = input_data["right"]
        if self.last_state is not None and hand["hand_trigger"]:
            desired_pos = hand["pos"]
            last_pos, last_quat = self.last_state
            action[0:3] = (np.array(desired_pos) - np.array(last_pos)) * 100
            # d_rot = R.from_quat(desired_quat) * R.from_quat(last_quat).inv()
            # action[3:6] = d_rot.as_euler("xyz") * 5
            if hand["index_trigger"]:
                action[-1] = 10
            else:
                action[-1] = -10
        if not hand["hand_trigger"]:
            self.last_state = None
        else:
            self.last_state = (hand["pos"], hand["rot"])
        return action

    def stop(self):
        pass


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--env_name", type=str, default="basketball-v2")
    args = parser.parse_args()

    # Initialize MetaWorld Benchmark
    ml1 = metaworld.ML1(args.env_name)

    # # Create Environment
    if args.env_name not in ml1.train_classes:
        raise ValueError(
            f"Environment '{args.env_name}' is not available."
            f"Available environments: {list(ml1.train_classes.keys())}"
        )

    env = ml1.train_classes[args.env_name]()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    # Initialize MujocoPublisher
    MujocoPublisher(
        env.model,
        env.data,
        host=args.host,
        visible_geoms_groups=list(range(3)),
    )
    controller = MQ3CartController(MetaQuest3("IRLMQ3-1"))

    # Main Loop
    env.reset()
    env.max_path_length = 1000000000
    while True:
        action = controller.get_action()
        env.step(action)
        continue


if __name__ == "__main__":
    main()
