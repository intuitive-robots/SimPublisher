from typing import List

from alr_sim.sims.mj_beta import MjScene

from .mj_publisher import MujocoPublisher


class SFPublisher(MujocoPublisher):

    def __init__(
        self,
        sf_mj_sim: MjScene,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        super().__init__(
            sf_mj_sim.model,
            sf_mj_sim.data,
            host,
            no_rendered_objects,
            no_tracked_objects
        )
