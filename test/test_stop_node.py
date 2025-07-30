import argparse
import mujoco
import os
import time

from simpub.sim.mj_publisher import MujocoPublisher


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/xinkai/repository")
    # parser.add_argument("--name", type=str, default="unitree_h1")
    # parser.add_argument("--name", type=str, default="unitree_g1")
    parser.add_argument("--name", type=str, default="boston_dynamics_spot")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    xml_path = os.path.join(
        args.path, "mujoco_menagerie", args.name, "scene_arm.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    publisher = MujocoPublisher(
        model, data, args.host, visible_geoms_groups=list(range(1, 3))
    )
    for _ in range(100):
        mujoco.mj_step(model, data)
        time.sleep(0.01)
    publisher.shutdown()
