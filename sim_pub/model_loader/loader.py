import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
import abc
from typing import TypedDict, Dict, List, Union
import os
from json import dumps
from matplotlib.pyplot import cla
# import trimesh

from ..msg import MsgPack
from .ucomponent import *
from .utils import *

def tree_to_list_dfs(
        root: UGameObject,
        game_objects_list: list[UGameObject] = None,
    ):
    if game_objects_list is None:
        game_objects_list = list()
    for child in root.children:
        game_objects_list.append(child)
        tree_to_list_dfs(child, game_objects_list)
    return game_objects_list

class XMLLoader(abc.ABC):

    def __init__(
            self, 
            file_path: str,
        ):
        self.asset_lib: AssetLibrary = AssetLibrary()
    #     self.include_xml_file(file_path)
        self.file_path = file_path
        root_xml = self.get_root_from_xml_file(file_path)
        self.parse_xml(root_xml)

    def get_root_from_xml_file(self, file_path: str) -> XMLNode:
        file_path = os.path.abspath(file_path)
        assert os.path.exists(file_path), f"The file '{file_path}' does not exist."
        tree_xml = ET.parse(file_path)
        return tree_xml.getroot()


    @abc.abstractmethod
    def parse_xml(self, root: ET.Element) -> str:
        raise NotImplemented

    def generate_scene_msg(self) -> MsgPack:
        game_obj_list = tree_to_list_dfs(SceneRoot())
        scene_list = [obj.to_dict() for obj in game_obj_list]
        return MsgPack("Scene_Model", dumps(scene_list))





@singleton
class AssetLibrary:
    
    def __init__(self) -> None:
        self.asset_path: dict[str, str]
        
    def include_mesh(self, file_name, file_path):
        pass

    # TODO: Try to inplement it
    def include_texture(self):
        pass

    def include_dae_file(self, file_name, file_path):
        pass

    def load_asset(self, asset_id) -> str:
        pass




# if __name__ == "__main__":
#     model_pub = ModelPublisher()
#     model_pub.include_mjcf_file("../panda.xml")
    
    # print(tree)
    # urdf = URDFLoader(tree)
    # print(tree.findall("worldbody"))
    # links = urdf.load_links()
    # for link in links:
    #     print(link)