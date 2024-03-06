import chunk
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
from typing import List, Dict, Callable, Optional, Tuple, TypeVar
from copy import deepcopy
from collections import OrderedDict

from .loader import XMLLoader
from .ucomponent import UGameObject, SceneRoot
from .ucomponent import UVisual, UVisualType
from .ucomponent import UComponent
from .ucomponent import UJoint
from .utils import *
from sim_pub.model_loader import ucomponent

def set_sphere_mjcf(geom: XMLNode, visual: UVisual) -> None:
    visual.type = UVisualType.SPHERE
    size = get_size(geom)
    visual.scale = [size[0], size[0], size[0]]
    pos = get_pos(geom)
    visual.pos = [-pos[1], pos[2], pos[0]]
    rot = get_rot(geom)
    visual.pos = [-rot[1], rot[2], rot[0]]
    if "fromto" in geom.attrib:
        pass
        # TODO: support for the fromto
    # TODO: other tags for pos and rot

    # TODO: create other shapes

def set_cylinder_mjcf(geom: XMLNode, visual: UVisual) -> None:
    visual.type = UVisualType.CYLINDER
    size = get_size(geom)
    visual.scale = [size[0] * 2, size[1], size[0] * 2]
    pos = get_pos(geom)
    visual.pos = [-pos[1], pos[2], pos[0]]
    rot = get_rot(geom)
    visual.pos = [-rot[1], rot[2], rot[0]]

def set_capsule_mjcf(geom: XMLNode, visual: UVisual) -> None:
    visual.type = UVisualType.CAPSULE
    size = get_size(geom)
    visual.scale = [size[0] * 2, size[1], size[0]]
    pos = get_pos(geom)
    visual.pos = [-pos[1], pos[2], pos[0]]
    rot = get_rot(geom)
    visual.pos = [-rot[1], rot[2], rot[0]]

def ros2unity(array: List[float]):
    return [-array[1], array[2], array[0]]
    

class MJCFLoader(XMLLoader):

    def __init__(self, file_path: str):
        self.tag_func_dict: Dict[str, Callable[[XMLNode, UGameObject], UGameObject]] = {
            "worldbody": self._parse_worldbody,
            "body": self._parse_body,
            "geom": self._parse_geom,
            "mesh": self._parse_mesh,
            "include": self._parse_include,
        }
        self.default_class_dict: OrderedDict[str, Dict[str, Dict[str, str]]] = OrderedDict()
        super().__init__(file_path)


    def parse_xml(self, root_xml: XMLNode) -> str:
        self.assembly_include_files(root_xml)
        # self.process_default_class(root_xml)
        # top_default_dict = self.default_class_dict[list(self.default_class_dict)[0]]
        # self.replace_class_defination(root_xml, top_default_dict)
        tree = ET.ElementTree(root_xml)
        xml_str = ET.tostring(root_xml, encoding='utf8', method='xml').decode()
        print(xml_str)
        # TODO: replace all the class tag
        # for worldbody in root_xml.findall("worldbody"):
            # self._parse_worldbody(worldbody)

    def assembly_include_files(self, root_xml: XMLNode) -> None:
        for child in root_xml:
            if child.tag != "include":
                self.assembly_include_files(child)
                continue
            sub_xml_path = os.path.join(self.file_path, "..", child.attrib["file"])
            sub_root_xml = self.get_root_from_xml_file(sub_xml_path)
            root_xml.extend(sub_root_xml)
            root_xml.remove(child)

    def process_default_class(self, xml: XMLNode, parent_dict: Dict[str, Dict[str, str]] = None):
        new_class_dict = {} if parent_dict is None else deepcopy(parent_dict)
        if xml.tag == "default":
            if "class" in xml.attrib:
                self.default_class_dict[xml.attrib["class"]] = new_class_dict
            for child in xml:
                if child.tag == "default":
                    continue
                if child.tag not in new_class_dict.keys():
                    new_class_dict[child.tag] = child.attrib
                else:
                    new_class_dict[child.tag].update(child.attrib)
        for default in xml.findall("default"):
            self.process_default_class(default, new_class_dict)
        
    def replace_class_defination(self, xml: XMLNode, parent_dict: Dict[str, Dict[str, str]]):
        if xml.tag == "default":
            return

        # default_class = xml.attrib["class"] if "class" in xml.attrib.keys() else default_class
        default_dict = self.default_class_dict[xml.attrib["class"]] if "class" in xml.attrib.keys() else parent_dict
            # default_class = self.default_class_dict[xml.attrib["class"]]
        # default_dict = self.default_class_dict[default_class]
        if xml.tag in default_dict:
            attrib = deepcopy(default_dict[xml.tag])
            attrib.update(xml.attrib)
            xml.attrib = attrib
        if "childclass" in xml.attrib:
            child_class = xml.attrib["childclass"]
            parent_dict = self.default_class_dict[child_class]
        for child in xml:
            self.replace_class_defination(child, parent_dict)
            # self.replace_class_defination(child, xml.attrib.get("childclass", default_class))

    def _parse(self, xml_element: XMLNode, parent: UGameObject) -> None:
        # parse itself
        if "class" in xml_element.attrib.keys():
            pass
        if xml_element.tag not in self.tag_func_dict.keys():
            return
        self.tag_func_dict[xml_element.tag](xml_element, parent)
        for xml_child in xml_element:
            self._parse(xml_child, )

    def _parse_include(self, parent: ET.Element) -> None:
        self.parse_xml()

    def _parse_asset(self, assets: XMLNode, parent: UGameObject) -> UGameObject:
        return parent

    def _parse_worldbody(self, worldbody: XMLNode):
        for body in worldbody.findall("body"):
            game_object = self._parse_body(body)
            self.root_object.add_child(game_object)

    def _parse_body(self, body: XMLNode, parent: UGameObject) -> UGameObject:
        # the basic data of body
        game_object = UGameObject(get_name(body), parent)
        game_object.pos = ros2unity(get_pos(body))
        game_object.rot = ros2unity(get_rot(body))
        game_object.set_parent(parent)
        return game_object

    def _parse_geom(self, geom: XMLNode, parent: UGameObject) -> UGameObject:
        geom_type = geom.get("type", "mesh")
        visual = UVisual(get_name(geom))
        if geom_type == "sphere":
            set_sphere_mjcf(geom, visual)
        elif geom_type == "box":
            visual.type = UVisualType.BOX
        elif geom_type == "capsule":
            set_capsule_mjcf(geom, visual)
        elif geom_type == "cylinder":
            set_cylinder_mjcf(geom, visual)
        elif geom_type == "mesh":
            pass
        parent.add_visual(visual)
        return parent

    def _parse_mesh(self, mesh: XMLNode, parent: UGameObject) -> UGameObject:
        if "name" not in mesh.attrib:
            mesh.set("name", mesh.get("file").rsplit(".", 1)[0])
        self.asset_lib.include_mesh(mesh.get("name"), mesh.get("file"))


