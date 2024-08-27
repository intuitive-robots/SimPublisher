from typing import List
from gym.envs.mujoco.mujoco_env import MujocoEnv
import fancy_gym
import os

from .mj_publisher import MujocoPublisher

FancyGymPath = os.path.dirname(fancy_gym.__file__)
FancyGymEnvPathDict = {
    "BoxPushingDense-v0":
        os.path.join(
            FancyGymPath,
            "envs/mujoco/box_pushing/assets/box_pushing.xml"
        )
}


class FancyGymPublisher(MujocoPublisher):

    def __init__(
        self,
        env_name: str,
        mj_env: MujocoEnv,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:

        super().__init__(
            mj_env.model,
            mj_env.data,
            FancyGymEnvPathDict[env_name],
            host,
            no_rendered_objects,
            no_tracked_objects,
        )
