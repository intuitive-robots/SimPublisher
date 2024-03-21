from __future__ import annotations
import abc
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
from typing import List, Dict, Callable, Optional, Tuple, TypeVar
from copy import deepcopy
from collections import OrderedDict
import trimesh

from .loader import XMLLoader
from .ucomponent import UGameObject, SceneRoot
from .ucomponent import UVisual, UVisualType
from .ucomponent import UComponent
from .ucomponent import UMesh, UMaterial
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

def set_plane_mjcf(geom: XMLNode, visual: UVisual) -> None:
    visual.type = UVisualType.PLANE
    size = get_size(geom)
    visual.scale = [size[0], size[1], 1]
    pos = get_pos(geom)
    visual.pos = [-pos[1], pos[2], pos[0]]
    rot = get_rot(geom)
    visual.pos = [-rot[1], rot[2], rot[0]]

def set_mesh_mjcf(geom: XMLNode, visual: UVisual) -> None:
    visual.type = UVisualType.MESH
    # visual.mesh = geom.get("file")
    pos = get_pos(geom)
    visual.pos = [-pos[1], pos[2], pos[0]]
    rot = get_rot(geom)
    visual.pos = [-rot[1], rot[2], rot[0]]

def ros2unity(array: List[float]):
    return [-array[1], array[2], array[0]]
    

class MJCFDefault:
    top: MJCFDefault = None
    def __init__(
            self,
            xml: XMLNode = None,
            parent: MJCFDefault = None,
        ) -> None:
        self._dict: Dict[str, Dict[str, str]] = dict()
        if parent is None:
            MJCFDefault.top = self
            self.class_name = "main"
        else:
            self.class_name: str = xml.attrib["class"]
            self.inherit_fron_parent(xml, parent)

    def update(self, xml: XMLNode) -> None:
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in self._dict.keys():
                self._dict[child.tag] = child.attrib
            else:
                self._dict[child.tag].update(child.attrib)

    @classmethod
    def get_top(cls) -> MJCFDefault:
        if cls.top is None:
            cls.top = MJCFDefault()
        return cls.top

    def inherit_fron_parent(
            self,
            xml: XMLNode,
            parent: MJCFDefault,
        ) -> None:

        default_dict: Dict[str, Dict[str, str]] = deepcopy(parent._dict)       
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in default_dict.keys():
                default_dict[child.tag] = deepcopy(child.attrib)
            else:
                default_dict[child.tag].update(child.attrib)
        self._dict = default_dict

    def replace_class_attrib(self, xml: XMLNode) -> None:
        if xml.tag in self._dict:
            attrib = deepcopy(self._dict[xml.tag])
            attrib.update(xml.attrib)
            xml.attrib = attrib
        return None

class MJCFLoader(XMLLoader):

    def __init__(self, file_path: str):
        self.tag_func_dict: Dict[str, Callable[[XMLNode, UGameObject], UGameObject]] = {
            "worldbody": self._parse_worldbody,
            "body": self._parse_body,
            "geom": self._parse_geom,
            "mesh": self._parse_mesh,
        }
        self.default_dict: Dict[str, MJCFDefault] = {}
        super().__init__(file_path)


    def parse_xml(self, root_xml: XMLNode) -> str:
        self.assembly_include_files(root_xml)
        self.import_default(root_xml)
        self.replace_class_defination(root_xml, MJCFDefault.get_top())
        self._parse(root_xml, self.root_object)
        # for game_object in self.game_object_dict.values():
        #     print(game_object.to_dict())
        tree = ET.ElementTree(root_xml)
        xml_str = ET.tostring(root_xml, encoding='utf8', method='xml').decode()
        # print(xml_str)

    def assembly_include_files(self, root_xml: XMLNode) -> None:
        for child in root_xml:
            if child.tag != "include":
                self.assembly_include_files(child)
                continue
            sub_xml_path = os.path.join(self.file_path, "..", child.attrib["file"])
            sub_root_xml = self.get_root_from_xml_file(sub_xml_path)
            root_xml.extend(sub_root_xml)
            root_xml.remove(child)

    def import_default(self, root_xml: XMLNode) -> None:
        for xml in root_xml:
            if xml.tag == "default":
                self._parse_default(xml)
            else:
                self.import_default(xml)
        default_xmls = root_xml.findall('default')
        for default_xml in default_xmls:
            root_xml.remove(default_xml)
        
    def replace_class_defination(self, xml: XMLNode, parent: MJCFDefault):
        if xml.tag == "default":
            return
        default = self.default_dict[xml.attrib["class"]] if "class" in xml.attrib.keys() else parent
        default.replace_class_attrib(xml)
        parent = self.default_dict[xml.attrib["childclass"]] if "childclass" in xml.attrib else parent
        for child in xml:
            self.replace_class_defination(child, parent)


    def _parse(self, xml_element: XMLNode, parent: UGameObject) -> None:
        if xml_element.tag in self.tag_func_dict.keys():
            parent = self.tag_func_dict[xml_element.tag](xml_element, parent)
        for xml_child in xml_element:
            # print(xml_child.tag, parent)
            self._parse(xml_child, parent)

    def _parse_include(self, parent: ET.Element) -> None:
        self.parse_xml()

    def _parse_default(self, default_xml: XMLNode, parent: MJCFDefault = None) -> None:
        if parent is None:
            default = MJCFDefault.get_top()
            default.update(default_xml)
        else:
            default = MJCFDefault(default_xml, parent)
            self.default_dict[default.class_name] = default
        for default_child_xml in default_xml.findall("default"):
            self._parse_default(default_child_xml, default)
        return 

    def _parse_asset(self, assets: XMLNode, parent: UGameObject) -> UGameObject:
        for asset in assets:
            if asset.tag == "mesh":
                self._parse_mesh(asset, parent)
            elif asset.tag == "material":
                self._parse_material(asset, parent)
            else:
                print("Unknown asset type")
        return parent

    def _parse_worldbody(self, worldbody: XMLNode, parent: UGameObject) -> UGameObject:
        return parent

    def _parse_body(self, body: XMLNode, parent: UGameObject) -> UGameObject:
        # the basic data of body
        game_object = UGameObject(get_name(body), parent)
        game_object.pos = ros2unity(get_pos(body))
        game_object.rot = ros2unity(get_rot(body))
        self.game_object_dict[game_object.name] = game_object
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
        elif geom_type == "plane":
            set_plane_mjcf(geom, visual)
        elif geom_type == "mesh":
            set_mesh_mjcf(geom, visual)
        else:
            raise ValueError(f"Unknown geom type: {geom_type}")
        parent.add_visual(visual)
        return parent

    def _parse_mesh(self, mesh: XMLNode, parent: UGameObject) -> UGameObject:
        mesh_name, mesh_type = mesh.get("file").rsplit(".", 1)
        if "name" not in mesh.attrib:
            mesh.set("name", mesh_name)
        if mesh_type == "stl":
            self.asset_lib.include_stl(mesh_name, mesh_type)
        elif mesh_type == "obj":
            self.asset_lib.include_obj(mesh_name, mesh_type)
        else:
            print("Unknown mesh type")
        return parent

    def _parse_material(self, material: XMLNode, parent: UGameObject) -> UGameObject:
        # assert "name" in material.attrib, "The material must have a name."
        # u_material = UMaterial(material.get("name"))
        # u_material.ambient = material.get("ambient", )
        # u_material.glossiness
        return parent
