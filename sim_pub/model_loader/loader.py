import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
import abc
from typing import TypedDict, Dict, List, Union
import os
from json import dumps
from matplotlib.pyplot import cla
import trimesh
from trimesh.base import Trimesh

from ..server.msg import MsgPack
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
            xml_path: str,
        ):
        self.xml_path = xml_path
        self.xml_floder = os.path.abspath(os.path.join(self.xml_path, ".."))
        self.asset_lib: AssetLibrary = AssetLibrary(self.xml_floder)
        self.root_object = SceneRoot()
        self.game_object_dict: Dict[str, UGameObject] = {
            "SceneRoot": SceneRoot(),
        }
        root_xml = self.get_root_from_xml_file(xml_path)
        self.parse_xml(root_xml)

    def get_root_from_xml_file(self, xml_path: str) -> XMLNode:
        xml_path = os.path.abspath(xml_path)
        assert os.path.exists(xml_path), f"The file '{xml_path}' does not exist."
        tree_xml = ET.parse(xml_path)
        return tree_xml.getroot()

    @abc.abstractmethod
    def parse_xml(self, root: ET.Element) -> str:
        raise NotImplemented

    def generate_scene_msg(self) -> MsgPack:
        game_obj_list = tree_to_list_dfs(SceneRoot())
        scene_list = [obj.to_dict() for obj in game_obj_list]
        return MsgPack("Scene_Model", dumps(scene_list))

class Asset:
    def __init__(self, asset_id: str, file_path: str) -> None:
        self.id = asset_id
        self.file_path = file_path

    @abc.abstractclassmethod
    def load_asset(self) -> str:
        pass
    
    def __str__(self) -> str:
        return f"Asset: {self.id} Path: {self.file_path}"

class STLAsset(Asset):
    def __init__(self, asset_id: str, file_path: str) -> None:
        super().__init__(asset_id, file_path)
        pass

    def load_asset(self) -> str:
        # trimesh.load(mesh.get("file"))
        pass
    
class OBJAsset(Asset):
    def __init__(self, asset_id: str, file_path: str) -> None:
        super().__init__(asset_id, file_path)
        pass

    def load_asset(self) -> str:
        obj_meshes: List[Trimesh] = trimesh.load_mesh(self.file_path)
        # for mesh in obj_meshes:
        #     mesh.

class AssetLibrary:
    
    def __init__(self, asset_path: str) -> None:
        self._assets: dict[str, Asset] = dict()
        self.asset_path = asset_path
        print(self.asset_path)

    def check_asset_path(self, file_path: str) -> str:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.asset_path, file_path)
        # print(self.asset_path, file_path)
        return file_path

    def include_stl(self, asset_id, file_name) -> None:
        file_path = self.check_asset_path(file_path)
        self._assets[asset_id] = STLAsset(asset_id, file_path)
        return

    def include_obj(self, asset_id, file_path) -> None:
        file_path = self.check_asset_path(file_path)
        self._assets[asset_id] = OBJAsset(asset_id, file_path)
        return
        
    # TODO: Try to implement it
    def include_texture(self):
        pass

    # TODO: Try to implement it
    def include_dae_file(self, file_name, file_path):
        pass

    def load_asset(self, asset_id) -> str:
        if asset_id in self._assets.keys():
            print(f"Asset {asset_id} already loaded.")
            return
        return self._assets[asset_id].load_asset()




# if __name__ == "__main__":
#     model_pub = ModelPublisher()
#     model_pub.include_mjcf_file("../panda.xml")
    
    # print(tree)
    # urdf = URDFLoader(tree)
    # print(tree.findall("worldbody"))
    # links = urdf.load_links()
    # for link in links:
    #     print(link)