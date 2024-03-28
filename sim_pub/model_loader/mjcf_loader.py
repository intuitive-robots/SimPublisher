from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
from typing import List, Dict, Callable
from copy import deepcopy

from .loader import XMLLoader
from .ucomponent import UGameObject, SceneRoot
from .ucomponent import UVisual, UVisualType
from .ucomponent import UMesh, UMaterial
from .utils import *
from sim_pub.model_loader import ucomponent

class MJCFDefault:
    Top: MJCFDefault = None
    def __init__(
            self,
            xml: XMLNode = None,
            parent: MJCFDefault = None,
        ) -> None:
        self._dict: Dict[str, Dict[str, str]] = dict()
        self.class_name = "main"
        if parent is None:
            MJCFDefault.Top = self
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

    # @classmethod
    # def top(cls) -> MJCFDefault:
    #     if cls._top is None:
    #         cls._top = MJCFDefault()
    #     return cls._top

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
    GeomTypeMap: Dict[str, UVisualType] = {
        "box": UVisualType.CUBE,
        "sphere": UVisualType.SPHERE,
        "cylinder": UVisualType.CYLINDER,
        "capsule": UVisualType.CAPSULE,
        "plane": UVisualType.PLANE,
        "mesh": UVisualType.MESH,
    }
    GeomSizeMap: Dict[str, Callable[[list[float]], List[float]]] = {
        "box": lambda size : [size[1] * 2, size[2] * 2, size[0] * 2],
        "sphere": lambda size : [size[0], size[0], size[0]],
        "cylinder": lambda size : [size[0] * 2, size[1], size[0] * 2],
        "capsule": lambda size : [size[0] * 2, size[1], size[0]],
        "plane": lambda _ : [1.0, 1.0, 1.0],
        "mesh": lambda _ : [1.0, 1.0, 1.0],
    }

    def __init__(self, file_path: str):
        self.tag_func_dict: Dict[str, Callable[[XMLNode, UGameObject], UGameObject]] = {
            "worldbody": self._parse_worldbody,
            "body": self._parse_body,
            "geom": self._parse_geom,
        }
        self.default_dict: Dict[str, MJCFDefault] = {}
        super().__init__(file_path)

    def parse_xml(self, root_xml: XMLNode) -> str:
        self.assembly_include_files(root_xml)
        top_default = self.export_default(root_xml)
        self.import_default(root_xml, top_default)
        self.load_asset(root_xml)
        self._parse(root_xml, self.root_object)
        # for game_object in self.game_object_dict.values():
        #     print(game_object.to_dict())
        # tree = ET.ElementTree(root_xml)
        # xml_str = ET.tostring(root_xml, encoding='utf8', method='xml').decode()
        # print(xml_str)

    def assembly_include_files(self, root_xml: XMLNode) -> None:
        for child in root_xml:
            if child.tag != "include":
                self.assembly_include_files(child)
                continue
            sub_xml_path = os.path.join(self.xml_path, "..", child.attrib["file"])
            sub_root_xml = self.get_root_from_xml_file(sub_xml_path)
            root_xml.extend(sub_root_xml)
            root_xml.remove(child)

    def export_default(self, root_xml: XMLNode) -> MJCFDefault:
        for xml in root_xml:
            if xml.tag == "default":
                self._parse_default(xml)
            else:
                self.export_default(xml)
        default_xmls = root_xml.findall('default')
        for default_xml in default_xmls:
            root_xml.remove(default_xml)
        return MJCFDefault.Top

    def _parse_default(self, default_xml: XMLNode, parent: MJCFDefault = None) -> None:
        if parent is None:
            default = MJCFDefault()
            default.update(default_xml)
        else:
            default = MJCFDefault(default_xml, parent)
            self.default_dict[default.class_name] = default
        for default_child_xml in default_xml.findall("default"):
            self._parse_default(default_child_xml, default)
        return 

    def import_default(self, xml: XMLNode, parent: MJCFDefault) -> None:
        if xml.tag == "default":
            return
        default = self.default_dict[xml.attrib["class"]] if "class" in xml.attrib.keys() else parent
        default.replace_class_attrib(xml)
        parent = self.default_dict[xml.attrib["childclass"]] if "childclass" in xml.attrib else parent
        for child in xml:
            self.import_default(child, parent)
        return

    def load_asset(self, root_xml: XMLNode) -> None:
        asset_xml = root_xml.findall("asset")
        for assets in asset_xml:
            for asset in assets:
                if asset.tag == "mesh":
                    self._parse_mesh(asset)
                elif asset.tag == "material":
                    self._parse_material(asset)
                else:
                    print(f"Warning: Unsupported asset type '{asset.tag}'")
        return 

    def _parse_mesh(self, mesh: XMLNode) -> None:
        mesh_file = mesh.get("file")
        mesh_name, mesh_type = mesh_file.rsplit(".", 1)
        if "name" not in mesh.attrib:
            mesh.set("name", mesh_name)
        if mesh_type == "stl":
            self.asset_lib.include_stl(mesh_name, mesh_file)
        elif mesh_type == "obj":
            self.asset_lib.include_obj(mesh_name, mesh_file)
        else:
            print("Warning: Unknown mesh type {mesh_type}")
        return 

    def _parse_material(self, material: XMLNode) -> None:        
        assert "name" in material.attrib, "The material must have a name."
        # u_material = UMaterial(material.attrib["name"])
        # u_material.ambient = material.get("ambient", )
        # u_material.glossiness
        return 

    def _parse(self, xml_element: XMLNode, parent: UGameObject) -> None:
        if xml_element.tag in self.tag_func_dict.keys():
            parent = self.tag_func_dict[xml_element.tag](xml_element, parent)
        for xml_child in xml_element:
            # print(xml_child.tag, parent)
            self._parse(xml_child, parent)
        return 

    def _parse_worldbody(self, worldbody: XMLNode, parent: UGameObject) -> UGameObject:
        return parent

    def _parse_body(self, body: XMLNode, parent: UGameObject) -> UGameObject:
        # TODO: As name tag is not mendatory, we need to generate a unique name for each body,
        # TODO: and the name should be the key of the Mujoco Simulation.
        assert "name" in body.attrib, "The body must have a name."
        game_object = UGameObject(body.get("name"), parent)
        game_object.pos = ros2unity(extract_array_from_xml(body, "pos"))
        game_object.rot = ros2unity(extract_array_from_xml(body, "rot"))
        self.game_object_dict[game_object.name] = game_object
        return game_object

    def _parse_geom(self, geom: XMLNode, parent: UGameObject) -> UGameObject:
        visual_name = geom.get("name", f"{parent.name}_visual_{len(parent.visual)}")
        visual = UVisual(visual_name)
        visual.pos = ros2unity(extract_array_from_xml(geom, "pos"))
        visual.rot = ros2unity(extract_array_from_xml(geom, "rot"))
        assert "type" in geom.attrib, "The geom must have a type."
        assert visual.type not in MJCFLoader.GeomTypeMap.keys(), "The geom type is not supported."
        visual.type = MJCFLoader.GeomTypeMap[geom.attrib["type"]]
        geom_size = extract_array_from_xml(geom, "size", "1.0 1.0 1.0")
        visual.scale = ros2unity(MJCFLoader.GeomSizeMap[geom.attrib["type"]](geom_size))
        parent.add_visual(visual)
        return parent

