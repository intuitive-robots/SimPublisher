
from xml.etree.ElementTree import Element as XMLNode
import numpy as np
from typing import List, Dict
import xml.etree.ElementTree as ET
import os
from os.path import join as pjoin

from alr_sim.sims.mj_beta import MjScene
from alr_sim.sims.mj_beta.mj_utils.mj_scene_parser import MjSceneParser
from alr_sim.utils.sim_path import sim_framework_path

from simpub.parser.mjcf import MJCFParser, MJCFScene
from ..core.simpub_server import SimPublisher
from .mj_publisher import MujocoPublisher
from ..core.log import logger


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
