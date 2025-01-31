import argparse

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored
import robocasa.models.scenes.scene_registry
from termcolor import colored

from simpub.sim.robocasa_publisher import RobocasaPublisher

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--layout", type=int, default=2)
    parser.add_argument("--style", type=int, default=2)
    parser.add_argument("--robot", type=str, default="PandaOmron")
    args = parser.parse_args()

    # Create argument configuration
    config = {
        "env_name": "PnPCounterToCab",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "translucent_robot": False,
    }
    print(colored("Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )

    # Grab reference to controller config and convert it to json-encoded string

    env.layout_and_style_ids = [[args.layout, args.style]]
    env.reset()
    env.render()

    publisher = RobocasaPublisher(env, args.host)

    while True:
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
        env.render()