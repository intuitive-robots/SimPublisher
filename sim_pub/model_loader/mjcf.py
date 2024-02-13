import xml.etree.ElementTree as ET

from .loader import XMLFileLoader
from .unity_entity import UGameObject, SceneRoot, EmptyGameObject, UVisual, UJoint


class MJCFLoader(XMLFileLoader):

    def parse(self):
        for worldbody in self.root.findall("worldbody"):
            self.parse_worldbody(worldbody)

    def parse_worldbody(self, worldbody: ET.ElementTree):
        for body in worldbody.findall("body"):
            game_object = self.parse_object(body)
            game_object.parent = self.root_object


    def parse_object(self, object_element: ET.ElementTree) -> UGameObject:
        game_object = UGameObject(object_element)
        game_object.visual = self.parse_visual(object_element.find("gemo"))
        for body in object_element.findall("body"):
            child = self.parse_object(body)
            child.parent = game_object
            game_object.children.append(child)
        return game_object


    def parse_visual(self, link: ET.ElementTree) -> UVisual:
        pass


    def parse_joint(self, link: ET.ElementTree) -> UJoint:
        pass