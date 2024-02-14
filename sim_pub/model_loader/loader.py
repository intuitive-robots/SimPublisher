import xml.etree.ElementTree as ET
import abc
from typing import TypedDict, Dict, List, Union
import os
import trimesh

from sim_pub.base import ServerBase
from sim_pub.model_loader.unity import *
from utils import singleton

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

class URDFLoader(XMLFileLoader):
    pass

# @singleton
# class AssetLibrary:
    
#     def __init__(self) -> None:
#         self.asset: dict[str, List[float]]
        
#     def include_stl_file(self, file_path):
#         pass

#     def include_dae_file(self, file_path):
#         pass

#     def convert_mesh(mesh : trimesh.base.Trimesh, matrix : np.ndarray, name : str) -> UMesh:


#         verts = np.around(mesh.vertices, decimals=5) # load vertices with max 5 decimal points
#         verts[:, 0] *= -1 # reverse x pos of every vertex
#         verts = verts.tolist()

#         norms = np.around(mesh.vertex_normals, decimals=5) # load normals with max 5 decimal points
#         norms[:, 0] *= -1 # reverse x pos of every normal
#         norms = norms.tolist()

#         indices = mesh.faces[:, [2, 1, 0]].flatten().tolist() # reverse winding order 

#         rot, pos = decompose_transform_matrix(matrix) # decompose matrix 

#         # this needs to be tested
#         pos = [-pos[1], pos[2], -pos[0]]
#         rot = [-rot[1] - math.pi / 2, -rot[2], math.pi / 2 - rot[0] ]
#         scale = [1, 1, 1]

#         return UMesh(
#             name=name, 
#             position=pos, 
#             rotation=rot, 
#             scale=scale, 
#             indices=indices, 
#             vertices=verts, 
#             normals=norms, 
#             material=convert_material(mesh.visual.material)
#         )

class ModelPublisher:
    def __init__(self) -> None:
        self.xml_file_loaders : List[XMLFileLoader] = list()
        # self.asset_lib = AssetLibrary()

    def include_urdf_file(self, file_path):
        self.xml_file_loaders.append(URDFLoader(file_path))

    def include_mjcf_file(self, file_path):
        self.xml_file_loaders.append(MJCFLoader(file_path))

    def publish_models(self, server: ServerBase) -> None:
        game_obj_list = tree_to_list_dfs(SceneRoot())
        
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