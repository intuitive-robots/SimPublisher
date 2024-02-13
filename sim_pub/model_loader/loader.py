import xml.etree.ElementTree as ET
import abc
from typing import TypedDict, Dict, List, Union
import os
from queue import Queue

from sim_pub.base import ServerBase
from sim_pub.model_loader.unity_entity import *

def tree_to_list_dfs(
        root: UGameObject, 
        game_objects_list: list[UGameObject],
    ):
    for child in root.children:
        game_objects_list.append(child)
        tree_to_list_dfs(child)
    game_objects_list.append(EmptyGameObject)
    return game_objects_list


class XMLFileLoader(abc.ABC):

    def __init__(
            self, 
            file_path: str, 
            root: UGameObject = SceneRoot.instance()
        ):
        # file_path = os.path.abspath(file_path)
        assert os.path.exists(file_path), f"The file '{file_path}' does not exist."
        self.file_path = file_path
        self.root_object = root
        self.namespace = file_path
        self.parse()

    def parse(self):
        pass




class AssetLibrary:
    
    def __init__(self) -> None:
        self.asset: dict[str, List[float]]
        
    def include_stl_file(self, file_path):
        pass

    def include_dae_file(self, file_path):
        pass


class ModelPublisher:
    def __init__(self) -> None:
        self.xml_file_loaders : List[XMLFileLoader] = list()

    def include_urdf_file(self, file_path):
        self.xml_file_loaders.append(URDFLoader(file_path))

    def include_mjcf_file(self, file_path):
        self.xml_file_loaders.append(MJCFLoader(file_path))

    def publish_models(self, server: ServerBase) -> None:
        pass
        # server.send_str_msg()
        # create new publishers and assign them to server

if __name__ == "__main__":
    model_pub = ModelPublisher()
    model_pub.include_mjcf_file("../panda.xml")
    
    # print(tree)
    # urdf = URDFLoader(tree)
    # print(tree.findall("worldbody"))
    # links = urdf.load_links()
    # for link in links:
    #     print(link)