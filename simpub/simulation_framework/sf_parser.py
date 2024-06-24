from typing import List
import xml.etree.ElementTree as ET
from os.path import join as pjoin

from alr_sim.sims.mj_beta import MjScene
from alr_sim.sims.mj_beta.mj_utils.mj_scene_parser import MjSceneParser
from alr_sim.utils.sim_path import sim_framework_path
from xml.etree.ElementTree import Element as XMLNode
from ..mjcf.mjcf_parser import MJCFParser
from simpub.mjcf.mjcf_parser import MJCFScene


class SFParser(MJCFParser):
    def __init__(self, mj_sim: MjScene):
        self.sf_mj_scene_parser: MjSceneParser = mj_sim.mj_scene_parser
        super().__init__("")
        self._path = sim_framework_path("models", "mj", "surroundings")
        self._use_degree = False
        self._meshdir = "assets"
        self._texturedir = "textures"

    def parse(
        self,
        no_rendered_objects: List[str] = None,
    ) -> MJCFScene:
        if no_rendered_objects is None:
            no_rendered_objects = []
        self.no_rendered_objects = no_rendered_objects
        # print(self.sf_mj_scene_parser.mj_xml_string)
        raw_xml = ET.fromstring(self.sf_mj_scene_parser.mj_xml_string)
        return self._parse_xml(raw_xml)

    def _load_compiler(self, xml: XMLNode) -> None:

        for compiler in xml.findall("./compiler"):
            self._use_degree = (
                True if compiler.get("angle", "degree") == "degree" else False
            )
            self._eulerseq = compiler.get("eulerseq", "xyz")
            self._assetdir = sim_framework_path("models", "mj", "robot", "assets")

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
        print(f"assetdir: {self._assetdir}")
        print(f"meshdir: {self._meshdir}")
        print(f"texturedir: {self._texturedir}")
