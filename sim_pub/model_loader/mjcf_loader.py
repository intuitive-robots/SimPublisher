import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
from typing import List, Optional, Tuple, TypeVar

from .loader import XMLFileLoader, SceneImporter
from .ucomponent import UGameObject, SceneRoot 
from .ucomponent import UVisual, UVisualType
from .ucomponent import UJoint
from .utils import *

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
    

class MJCFLoader(XMLFileLoader):

    def parse(self, root: ET.Element) -> str:
        for worldbody in root.findall("worldbody"):
            self.parse_worldbody(worldbody)

    def parse_worldbody(self, worldbody: XMLNode):
        for body in worldbody.findall("body"):
            game_object = self.parse_body(body)
            self.root_object.add_child(game_object)

    def parse_body(self, body: XMLNode, parent: UGameObject = SceneRoot()) -> UGameObject:
        # the basic data of body
        game_object = UGameObject(get_name(body), parent)
        game_object.pos = ros2unity(get_pos(body))
        game_object.rot = ros2unity(get_rot(body))
        # generate the children
        for geom in body.findall("geom"):
            game_object.add_visual(self.parse_geom(geom))
        for body in body.findall("body"):
            child = self.parse_body(body, game_object)
            child.parent = game_object
            game_object.add_child(child)
        return game_object

    def parse_geom(self, geom: XMLNode) -> UVisual:
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
        return visual

    def parse_mesh(self, link: XMLNode) -> UVisual:
        pass

    def parse_joint(self, link: XMLNode) -> UJoint:
        pass


class MJCFImporter(SceneImporter):
    def include_xml_file(self, file_path: str) -> None:
        self.xml_file_loaders.append(MJCFLoader(file_path))