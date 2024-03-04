import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element as XMLNode
import abc
from typing import TypedDict, Dict, List, Union
import os
from json import dumps
import trimesh

from ..base import ServerBase
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

class XMLFileLoader(abc.ABC):

    def __init__(
            self, 
            file_path: str, 
            root: UGameObject = SceneRoot()
        ):
        # file_path = os.path.abspath(file_path)
        assert os.path.exists(file_path), f"The file '{file_path}' does not exist."
        tree = ET.parse(file_path)
        tree_root = tree.getroot()
        self.file_path = file_path
        self.root_object = root
        self.namespace = file_path
        self.parse(tree_root)

    @abc.abstractmethod
    def parse(self, root: ET.Element) -> str:
        raise NotImplemented

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
        game_object.pos = get_pos(body)
        game_object.rot = get_rot(body)
        # generate the children
        # for geom in body.findall("geom"):
        #     self.parse_geom(geom, game_object)
        for body in body.findall("body"):
            child = self.parse_body(body, game_object)
            child.parent = game_object
            game_object.add_child(child)
        return game_object

    def parse_geom(self, geom: XMLNode, game_object: UGameObject) -> None:
        geom_type = geom.get("type", "mesh")
        visual = UVisual(get_name(geom))
        if geom_type == "sphere":
            self.create_sphere(geom, visual)
        elif geom_type == "box":
            visual.type = UVisualType.BOX
        elif geom_type == "capsule":
            visual.type = UVisualType.CAPSULE
        elif geom_type == "mesh":
            pass
        game_object.add_visual(visual)

    def parse_mesh(self, link: XMLNode) -> UVisual:
        pass

    def parse_joint(self, link: XMLNode) -> UJoint:
        pass

    @staticmethod
    def create_sphere(geom: XMLNode, visual: UVisual) -> None:
        visual.type = UVisualType.SPHERE
        if "size" in geom.attrib:
            visual.scale = [float(geom["size"]), float(geom["size"]), float(geom["size"])]
        pos = get_pos(geom)
        visual.pos = [pos[1], -pos[2], pos[0]]
        rot = get_rot(geom)
        visual.pos = [rot[1], -rot[2], rot[0]]
        if "fromto" in geom.attrib:
            pass
            # TODO: support for the fromto
        # TODO: other tags for pos and rot

    # TODO: create other shapes

class URDFLoader(XMLFileLoader):
    pass

@singleton
class AssetLibrary:
    
    def __init__(self) -> None:
        self.asset_path: dict[str, str]
        
    def include_stl_file(self, file_name, file_path):
        pass

    def include_dae_file(self, file_name, file_path):
        pass

    def load_asset(self, asset_id) -> str:
        pass

    # def convert_mesh(mesh : trimesh.base.Trimesh, matrix : np.ndarray, name : str) -> UMesh:


    #     verts = np.around(mesh.vertices, decimals=5) # load vertices with max 5 decimal points
    #     verts[:, 0] *= -1 # reverse x pos of every vertex
    #     verts = verts.tolist()

    #     norms = np.around(mesh.vertex_normals, decimals=5) # load normals with max 5 decimal points
    #     norms[:, 0] *= -1 # reverse x pos of every normal
    #     norms = norms.tolist()

    #     indices = mesh.faces[:, [2, 1, 0]].flatten().tolist() # reverse winding order 

    #     rot, pos = decompose_transform_matrix(matrix) # decompose matrix 

    #     # this needs to be tested
    #     pos = [-pos[1], pos[2], -pos[0]]
    #     rot = [-rot[1] - math.pi / 2, -rot[2], math.pi / 2 - rot[0] ]
    #     scale = [1, 1, 1]

    #     return UMesh(
    #         name=name, 
    #         position=pos, 
    #         rotation=rot, 
    #         scale=scale, 
    #         indices=indices, 
    #         vertices=verts, 
    #         normals=norms, 
    #         material=convert_material(mesh.visual.material)
    #     )

@singleton
class SceneLoader:
    def __init__(self) -> None:
        self.xml_file_loaders : List[XMLFileLoader] = list()
        self.asset_lib = AssetLibrary()

    def include_urdf_file(self, file_path):
        self.xml_file_loaders.append(URDFLoader(file_path))

    def include_mjcf_file(self, file_path):
        self.xml_file_loaders.append(MJCFLoader(file_path))

    def generate_scene_msg(self) -> MsgPack:
        game_obj_list = tree_to_list_dfs(SceneRoot())
        scene_list = [obj.to_dict() for obj in game_obj_list]
        return MsgPack("Scene_Model", dumps(scene_list))
        # server.send_str_msg()
        # create new publishers and assign them to server

# if __name__ == "__main__":
#     model_pub = ModelPublisher()
#     model_pub.include_mjcf_file("../panda.xml")
    
    # print(tree)
    # urdf = URDFLoader(tree)
    # print(tree.findall("worldbody"))
    # links = urdf.load_links()
    # for link in links:
    #     print(link)