import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
import abc
from typing import TypedDict, Dict, List, Union
import os
from json import dumps
from matplotlib.pyplot import cla
import trimesh
from trimesh.base import Trimesh

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
        self.file_path = file_path
        self.root_object = SceneRoot()
        self.game_object_dict: Dict[str, UGameObject] = {
            "SceneRoot": SceneRoot(),
        }
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



class Asset:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.linked_objects: List[UGameObject] = list()

    @abc.abstractclassmethod
    def load_asset(self) -> str:
        pass

class STLAsset(Asset):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        pass

    def load_asset(self) -> str:
        # trimesh.load(mesh.get("file"))
        pass
    
class OBJAsset(Asset):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        pass

    def load_asset(self) -> str:
        obj_meshes: List[Trimesh] = trimesh.load_mesh(self.file_path)
        # for mesh in obj_meshes:
        #     mesh.


@singleton
class AssetLibrary:
    
    def __init__(self) -> None:
        self.asset_path: dict[str, Asset] = dict()
        
    def include_stl(self, asset_id, file_path) -> None:
        self.asset_path[asset_id] = STLAsset(file_path)
        return

    def include_obj(self, asset_id, file_path) -> None:
        self.asset_path[asset_id] = OBJAsset(file_path)
        return
        
    # TODO: Try to implement it
    def include_texture(self):
        pass

    def include_dae_file(self, file_name, file_path):
        pass

    def load_asset(self, asset_id) -> str:
        for asset in self.asset_path.values():
            asset.load_asset()




# if __name__ == "__main__":
#     model_pub = ModelPublisher()
#     model_pub.include_mjcf_file("../panda.xml")
    
    # print(tree)
    # urdf = URDFLoader(tree)
    # print(tree.findall("worldbody"))
    # links = urdf.load_links()
    # for link in links:
    #     print(link)