from typing import List
from gym.envs.mujoco.mujoco_env import MujocoEnv

from .mj_publisher import MujocoPublisher


class FancyGymPublisher(MujocoPublisher):

    def __init__(
        self,
        mj_env: MujocoEnv,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:

        super().__init__(
            mj_env.model,
            mj_env.data,
            host,
            no_rendered_objects,
            no_tracked_objects,
        )
