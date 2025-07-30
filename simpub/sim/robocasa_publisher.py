from typing import List, Optional

from robosuite.environments.base import MujocoEnv

from .mj_publisher import MujocoPublisher


class RobocasaPublisher(MujocoPublisher):
    def __init__(
        self,
        env: MujocoEnv,
        host: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            env.sim.model._model,
            env.sim.data._data,
            host,
            no_rendered_objects,
            no_tracked_objects,
            visible_geoms_groups=list(range(1, 5)),
        )
