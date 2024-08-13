
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

class SFParser(MJCFParser):
    def __init__(self, mj_sim: MjScene):
        self.sf_mj_scene_parser: MjSceneParser = mj_sim.mj_scene_parser
        super().__init__("")
        self._path = sim_framework_path("models", "mj", "surroundings")
        self._use_degree = False
        self._meshdir = "assets"
        self._texturedir = "textures"
        self._mj_sim_assets = mj_sim.mj_scene_parser.assets

    def parse(
        self,
        no_rendered_objects: List[str] = None,
    ) -> MJCFScene:
        if no_rendered_objects is None:
            no_rendered_objects = []
        self.no_rendered_objects = no_rendered_objects
        raw_xml = ET.fromstring(self.sf_mj_scene_parser.mj_xml_string)
        return self._parse_xml(raw_xml)

    def _merge_includes(self, root_xml: XMLNode) -> XMLNode:
        for child in root_xml:
            if child.tag != "include":
                self._merge_includes(child)
                continue
            sub_xml_path = os.path.join(self._path, child.attrib["file"])
            if os.path.exists(sub_xml_path):
                sub_xml_root = self.get_root_from_xml_file(sub_xml_path)
            else:
                for name, asset in self._mj_sim_assets.items():
                    if name.endswith(child.attrib["file"]):
                        sub_xml_root = ET.fromstring(asset)
                        break
            root_xml.extend(sub_xml_root)
        for child in root_xml:
            if child.tag == "include":
                root_xml.remove(child)
        return root_xml

    def _load_compiler(self, xml: XMLNode) -> None:

        for compiler in xml.findall("./compiler"):
            self._use_degree = (
                True if compiler.get("angle", "degree") == "degree" else False
            )
            self._eulerseq = compiler.get("eulerseq", "xyz")
            self._assetdir = sim_framework_path(
                "models", "mj", "robot", "assets"
            )

            if "meshdir" in compiler.attrib:
                self._meshdir = pjoin(self._path, compiler.get("meshdir", ""))
            else:
                self._meshdir = self._assetdir
            if "texturedir" in compiler.attrib:
                self._texturedir = pjoin(
                    self._path, compiler.get("texturedir", "")
                )
            else:
                self._texturedir = self._assetdir
        logger.info(f"assetdir: {self._assetdir}")
        logger.info(f"meshdir: {self._meshdir}")
        logger.info(f"texturedir: {self._texturedir}")


class SFPublisher(MujocoPublisher):

    def __init__(
        self,
        sf_mj_sim: MjScene,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        self.parser = SFParser(sf_mj_sim)
        self.mj_data: MjScene = sf_mj_sim.data
        self.mj_model: MjScene = sf_mj_sim.model
        self.tracked_obj_trans: Dict[str, np.ndarray] = dict()
        SimPublisher.__init__(
            self,
            self.parser.parse(),
            host,
            no_rendered_objects,
            no_tracked_objects
        )
        self.scene_message = self.sim_scene.to_string()
        for child in self.sim_scene.root.children:
            self.set_update_objects(child)
