from __future__ import annotations
from xml.etree.ElementTree import Element as XMLNode
import xml.etree.ElementTree as ET
import copy
import os
from os.path import join as pjoin
from typing import List, Dict
from pathlib import Path
from simpub.parser.mjcf.asset_loader import MeshLoader, TextureLoader
from simpub.simdata import SimObject, SimScene
from simpub.simdata import SimMaterial, SimTransform
from simpub.simdata import SimVisual
from .utils import str2list, str2listabs, ros2unity
from .utils import get_rot_from_xml, scale2unity, TypeMap
from ...core.log import logger


class MJCFScene(SimScene):
    def __init__(self) -> None:
        super().__init__()
        self.xml_string: str = None


class MJCFDefault:
    def __init__(
        self,
        xml: XMLNode = None,
        parent: MJCFDefault = None,
    ) -> None:
        self._dict: Dict[str, Dict[str, str]] = dict()
        self.class_name = "main" if parent is None else xml.attrib["class"]
        if xml is not None:
            self.import_xml(xml)

    def import_xml(self, xml: XMLNode, parent: MJCFDefault = None) -> None:
        if parent is not None:
            self._dict = copy.deepcopy(parent._dict)
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in self._dict.keys():
                self._dict[child.tag] = copy.deepcopy(child.attrib)
            else:
                self._dict[child.tag].update(child.attrib)

    def update_xml(self, xml: XMLNode) -> None:
        if xml.tag in self._dict:
            attrib = copy.deepcopy(self._dict[xml.tag])
            attrib.update(xml.attrib)
            xml.attrib = attrib


class MJCFParser:
    def __init__(
        self,
        file_path: str,
    ):
        self._xml_path = os.path.abspath(file_path)
        self._path = os.path.abspath(os.path.join(self._xml_path, ".."))

    def parse(
        self,
        no_rendered_objects: List[str] = None,
    ) -> MJCFScene:
        if no_rendered_objects is None:
            no_rendered_objects = []
        self.no_rendered_objects = no_rendered_objects
        raw_xml = self.get_root_from_xml_file(self._xml_path)
        return self._parse_xml(raw_xml)

    def get_root_from_xml_file(self, xml_path: str) -> XMLNode:
        xml_path = os.path.abspath(xml_path)
        assert os.path.exists(xml_path), f"File '{xml_path}' does not exist."
        tree_xml = ET.parse(xml_path)
        return tree_xml.getroot()

    def _parse_xml(self, raw_xml: XMLNode) -> MJCFScene:
        mj_scene = MJCFScene()
        xml = self._merge_includes(raw_xml)
        self._load_compiler(xml)
        self._load_defaults(xml)
        self._load_assets(xml, mj_scene)
        self._load_worldbody(raw_xml, mj_scene)
        mj_scene.xml_string = ET.tostring(raw_xml, encoding="unicode")
        return mj_scene

    def _merge_includes(self, root_xml: XMLNode) -> XMLNode:
        for child in root_xml:
            if child.tag != "include":
                self._merge_includes(child)
                continue
            sub_xml_path = os.path.join(self._path, child.attrib["file"])
            if not os.path.exists(sub_xml_path):
                logger.warning(f"Warning: File '{sub_xml_path}' does not exist.")
                continue
            sub_xml_root = self.get_root_from_xml_file(sub_xml_path)
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
            self._assetdir = pjoin(self._path, compiler.get("assetdir", ""))
            if "meshdir" in compiler.attrib:
                self._meshdir = pjoin(self._path, compiler.get("meshdir", ""))
            else:
                self._meshdir = self._assetdir
            if "texturedir" in compiler.attrib:
                self._texturedir = pjoin(self._path, compiler.get("texturedir", ""))
            else:
                self._texturedir = self._assetdir
        logger.info(f"assetdir: {self._assetdir}")
        logger.info(f"meshdir: {self._meshdir}")
        logger.info(f"texturedir: {self._texturedir}")

    def _load_defaults(self, root_xml: XMLNode) -> None:
        default_dict: Dict[str, MJCFDefault] = dict()
        default_dict["main"] = MJCFDefault()
        # only start _loading default tags under the mujoco tag
        for default_child_xml in root_xml.findall("./default"):
            self._parse_default(default_child_xml, default_dict)
        # replace the class attribute with the default values
        self._import_default(root_xml, default_dict)

    def _parse_default(
        self,
        default_xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent: MJCFDefault = None,
    ) -> None:
        default = MJCFDefault(default_xml, parent)
        default_dict[default.class_name] = default
        for default_child_xml in default_xml.findall("default"):
            self._parse_default(default_child_xml, default_dict, default)

    def _import_default(
        self,
        xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent_name: str = "main",
    ) -> None:
        if xml.tag == "default":
            return
        default_name = (
            xml.attrib["class"] if "class" in xml.attrib.keys() else parent_name
        )
        default = default_dict[default_name]
        default.update_xml(xml)

        parent_name = (
            xml.attrib["childclass"] if "childclass" in xml.attrib else parent_name
        )
        for child in xml:
            self._import_default(child, default_dict, parent_name)

    def _load_assets(self, xml: XMLNode, mj_scene: MJCFScene) -> None:
        for assets in xml.findall("./asset"):
            for asset in assets:
                if asset.tag == "mesh":
                    asset_file = pjoin(self._meshdir, asset.attrib["file"])
                    scale = str2list(asset.get("scale", "1 1 1"))
                    mesh, bin_data = MeshLoader.from_file(
                        asset_file,
                        asset.get("name", asset.attrib["file"]),
                        scale,
                    )
                    mj_scene.meshes.append(mesh)
                    mj_scene.raw_data[mesh.dataHash] = bin_data

                elif asset.tag == "material":
                    color = str2list(asset.get("rgba", "1 1 1 1"))
                    emission = float(asset.get("emission", "0.0"))
                    emissionColor = [emission * c for c in color]
                    material = SimMaterial(
                        id=asset.get("name") or asset.get("type"),
                        color=color,
                        emissionColor=emissionColor,
                        specular=float(asset.get("specular", "0.5")),
                        shininess=float(asset.get("shininess", "0.5")),
                        reflectance=float(asset.get("reflectance", "0.0")),
                    )
                    material.texture = asset.get("texture", None)
                    material.texsize = str2list(asset.get("texrepeat", "1 1"))
                    mj_scene.materials.append(material)
                elif asset.tag == "texture":
                    name = asset.get("name") or asset.get("type")
                    builtin = asset.get("builtin", "none")
                    tint = str2list(asset.get("rgb1", "1 1 1"))
                    if builtin != "none":
                        texture, bin_data = TextureLoader.fromBuiltin(
                            name, builtin, tint
                        )
                    else:
                        asset_file = pjoin(self._texturedir, asset.attrib["file"])
                        byte_data = Path(asset_file).read_bytes()
                        texture, bin_data = TextureLoader.from_bytes(
                            name, byte_data, asset.get("type", "cube"), tint
                        )
                    mj_scene.textures.append(texture)
                    mj_scene.raw_data[texture.dataHash] = bin_data
                else:
                    raise RuntimeError("Invalid asset", asset.tag)

    def _load_worldbody(self, xml: XMLNode, mj_scene: MJCFScene) -> None:
        mj_scene.root = SimObject(name="root")
        for worldbody in xml.findall("./worldbody"):
            for geom in worldbody.findall("geom"):
                visual = self._load_visual(geom)
                if visual is not None:
                    mj_scene.root.visuals.append(visual)
            for body in worldbody.findall("./body"):
                self._load_body(body, mj_scene.root)

    def _load_visual(self, visual: XMLNode) -> SimVisual:
        visual_type = visual.get("type", "box")
        pos = ros2unity(str2list(visual.get("pos", "0 0 0")))
        rot = get_rot_from_xml(visual)
        size = str2listabs(visual.get("size", "0.1 0.1 0.1"))
        scale = scale2unity(size, visual_type)
        trans = SimTransform(pos=pos, rot=rot, scale=scale)
        return SimVisual(
            type=TypeMap[visual_type],
            trans=trans,
            mesh=visual.get("mesh"),
            material=visual.get("material"),
            color=str2list(visual.get("rgba", "1 1 1 1")),
        )

    def _load_body(
        self,
        body: XMLNode,
        parent: SimObject,
    ) -> None:
        name = body.get("name")

        if name in self.no_rendered_objects:
            return

        trans = SimTransform(
            pos=ros2unity(str2list(body.get("pos", "0 0 0"))),
            rot=get_rot_from_xml(body),
        )

        visuals: List[SimVisual] = list()
        for geom in body.findall("geom"):
            # geom group 3 is for collision
            if int(geom.get("group", "2")) == 3:
                continue
            visual = self._load_visual(geom)
            if visual is not None:
                visuals.append(visual)
        new_gameobject = SimObject(
            name=name,
            trans=trans,
            visuals=visuals,
        )
        parent.children.append(new_gameobject)
        for child in body.findall("body"):
            self._load_body(child, new_gameobject)
        return
