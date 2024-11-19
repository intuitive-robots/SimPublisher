import numpy as np

from robosuite.environments.base import MujocoEnv as RobosuiteEnv
import xml.etree.ElementTree as ET
import os

from simpub.xr_device.meta_quest3 import MetaQuest3 as MetaQuest3Sim
from simpub.sim.mj_publisher import MujocoPublisher


from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from robosuite.robots import ROBOT_CLASS_MAPPING


class RobosuitePublisher(MujocoPublisher):

    def __init__(self, env: RobosuiteEnv):
        super().__init__(
            env.sim.model._model,
            env.sim.data._data,
            visible_geoms_groups=[1, 2, 3, 4]
        )


if __name__ == "__main__":
    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configuration
    config = {
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }

    bddl_path = "/home//LIBERO/libero/libero/bddl_files/"
    bddl_dataset_name = "libero_10"
    bddl_name = "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl"
    bddl_file = os.path.join(bddl_path, bddl_dataset_name, bddl_name)
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = "single-arm-opposed"
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # reset the environment
    env.reset()
    publisher = RobosuitePublisher(env)
    while True:
        action = np.random.randn(env.robots[0].dof - 1)  # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display