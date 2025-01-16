from typing import List

import numpy as np
import tqdm
from ..simdata import SimObject, SimScene, SimTransform, SimVisual
from ..simdata import SimMaterial, SimTexture, SimMesh
from ..simdata import VisualType
from ..core.log import logger


def get_scene_obj_type_str(sim, obj_type_id: int):
    # TODO, MAKE THIS A DICT
    # Get obj type name from obj type id
    if obj_type_id == sim.sceneobject_shape:
        return "shape"
    if obj_type_id == sim.sceneobject_joint:
        return "joint"
    if obj_type_id == sim.sceneobject_graph:
        return "graph"
    if obj_type_id == sim.sceneobject_camera:
        return "camera"
    if obj_type_id == sim.sceneobject_light:
        return "light"
    if obj_type_id == sim.sceneobject_dummy:
        return "dummy"
    if obj_type_id == sim.sceneobject_proximitysensor:
        return "proximitysensor"
    if obj_type_id == sim.sceneobject_octree:
        return "octree"
    if obj_type_id == sim.sceneobject_pointcloud:
        return "pointcloud"
    if obj_type_id == sim.sceneobject_visionsensor:
        return "visionsensor"
    if obj_type_id == sim.sceneobject_forcesensor:
        return "forcesensor"
    if obj_type_id == sim.sceneobject_script:
        return "script"
    raise ValueError(f"Unknown object type id: {obj_type_id}")


def get_primitive_type_str(sim, primitive_shape_id):
    # TODO, MAKE THIS A DICT
    if primitive_shape_id == sim.primitiveshape_none:
        return "none"
    if primitive_shape_id == sim.primitiveshape_plane:
        return "plane"
    if primitive_shape_id == sim.primitiveshape_disc:
        return "disc"
    if primitive_shape_id == sim.primitiveshape_cuboid:
        return "cuboid"
    if primitive_shape_id == sim.primitiveshape_spheroid:
        return "spheroid"
    if primitive_shape_id == sim.primitiveshape_cylinder:
        return "cylinder"
    if primitive_shape_id == sim.primitiveshape_cone:
        return "cone"
    if primitive_shape_id == sim.primitiveshape_capsule:
        return "capsule"
    if primitive_shape_id == sim.primitiveshape_heightfield:
        return "heightfield"
    raise ValueError(f"Unknown primitive type: {primitive_shape_id}")


def get_bit_positions(number):
    bit_positions = []
    position = 0
    while number > 0:
        if number & 1:
            bit_positions.append(position)
        number >>= 1
        position += 1
    return bit_positions


def ungroup_compound_objects(sim, visual_layer_list):
    # Exhaust flag
    flag = True

    # Get all objects id in a list.
    objects_id_list = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)

    # Ungroup compound objects
    for idx in objects_id_list:

        # Check visualization
        visualize = False
        if visual_layer_list is not None:
            obj_layers = get_bit_positions(sim.getIntProperty(idx, "layer"))
            if bool(set(visual_layer_list) & set(obj_layers)):
                visualize = True

        # Check object type
        obj_type_id = sim.getObjectType(idx)
        is_shape = get_scene_obj_type_str(sim, obj_type_id) == "shape"

        # In case of a visual shape, Check if shape is compound
        if is_shape and visualize:
            result = sim.getShapeGeomInfo(idx)[0]
            is_compound = bool(set(get_bit_positions(result)) & {0})
            if is_compound:
                handles = sim.ungroupShape(idx)
                flag = False
    if not flag:
        ungroup_compound_objects(sim, visual_layer_list)


def get_objects_info_dict(sim, visual_layer_list=None,
                          name_as_key=False):
    # Ungroup compound objects
    ungroup_compound_objects(sim, visual_layer_list)

    objects_handle_list = (
        sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0))

    obj_info_dict = {}

    # tqdm progress bar
    for idx in tqdm.tqdm(objects_handle_list):
        obj_name = sim.getObjectAlias(idx)
        visualize = False
        if visual_layer_list is not None:
            obj_layers = get_bit_positions(sim.getIntProperty(idx, "layer"))
            if bool(set(visual_layer_list) & set(obj_layers)):
                visualize = True

        parent_id = str(sim.getObjectParent(idx))
        if parent_id == "-1":
            parent_id = "world"

        obj_type_id = sim.getObjectType(idx)
        obj_type_str = get_scene_obj_type_str(sim, obj_type_id)
        idx_str = str(idx)
        obj_info_dict[idx_str] = {"name": obj_name,
                                  "parent_id": parent_id,
                                  "type": obj_type_str,
                                  "visualize": visualize}

        # Check shape type, return types: int, int, list
        if obj_type_str == "shape" and visualize:
            result, pureType, dimensions = sim.getShapeGeomInfo(idx)
            primitive_type_str = get_primitive_type_str(sim, pureType)
            vertices, indices, normals = sim.getShapeMesh(idx)
            vertices = np.asarray(vertices).reshape(-1, 3)  # 5958
            indices = np.asarray(indices).reshape(-1, 3)  # 11904
            normals = np.asarray(normals).reshape(-1, 3)  # 35712
            ambient_diffuse = sim.getShapeColor(
                idx, None, sim.colorcomponent_ambient_diffuse)[1]
            diffuse = sim.getShapeColor(
                idx, None, sim.colorcomponent_diffuse)[1]
            specular = sim.getShapeColor(
                idx, None, sim.colorcomponent_specular)[1]
            emission = sim.getShapeColor(
                idx, None, sim.colorcomponent_emission)[1]
            transparency = sim.getShapeColor(
                idx, None, sim.colorcomponent_transparency)[1]
            auxiliary = sim.getShapeColor(
                idx, None, sim.colorcomponent_auxiliary)[1]

            obj_info_dict[idx_str].update({
                "shape_result": result,
                "primitive_type": primitive_type_str,
                "dimensions": dimensions,
                "shape_vertices": vertices,
                "shape_indices": indices,
                "shape_normals": normals,
                "color_ambient_diffuse": ambient_diffuse,
                "color_diffuse": diffuse,
                "color_specular": specular,
                "color_emission": emission,
                "color_transparency": transparency,
                "color_auxiliary": auxiliary,
            })

        # Get Transform info, absolute to world
        if parent_id == "world":
            pos = sim.getObjectPosition(idx, sim.handle_world)
            quat = sim.getObjectQuaternion(idx, sim.handle_world)  # fixme, use quaternion?
        else:
            pos = sim.getObjectPosition(idx, int(parent_id))
            quat = sim.getObjectQuaternion(idx, int(parent_id))
        obj_info_dict[idx_str].update({
            "pos": pos,
            "quat": quat
        })

    # Add a virtual root object
    obj_info_dict["world"] = {"name": "world",
                              "visualize": False,
                              "parent_id": -1,
                              "type": "root",
                              "pos": [0, 0, 0],
                              "quat": [0, 0, 0, 1]}

    if name_as_key:
        name_as_key_obj_info_dict = {}
        # todo, when alias repeat, add a counter to the name
        # swap key and id in the value
        for id, info in obj_info_dict.items():
            name = info.pop("name")
            info["id"] = id
            name_as_key_obj_info_dict[name] = info
        return name_as_key_obj_info_dict

    else:
        return obj_info_dict


def mj2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat: List[float]) -> List[float]:
    # return [quat[2], -quat[3], -quat[1], quat[0]] # Mujoco
    return [quat[1], -quat[2], -quat[0], quat[3]]  # Vrep


class CoppeliasSimParser:
    def __init__(self, coppelia_sim, visual_layer_list):
        self.sim_scene = None
        self.visual_layer_list = visual_layer_list
        self.parse_scene(coppelia_sim)
        self.sim_scene.process_sim_obj(self.sim_scene.root)

    def parse(self):
        return self.sim_scene

    def parse_scene(self, cs_sim) -> SimScene:
        """
        Parse CoppeliaSim scene to SimScene
        """

        # Create SimScene
        sim_scene = SimScene()
        self.sim_scene = sim_scene

        # Info dict
        info_dict_id_as_key = get_objects_info_dict(
            cs_sim, self.visual_layer_list, name_as_key=False)

        # Build the hierarchy tree
        body_hierarchy = {}
        for obj_id, obj_info in info_dict_id_as_key.items():
            sim_object = self.process_body(obj_id, obj_info)

            body_hierarchy[obj_id] = {
                "parent_id": obj_info["parent_id"],
                "sim_object": sim_object,
            }

        # create a tree structure from the body hierarchy
        for body_id, body_info in body_hierarchy.items():
            parent_id = body_info["parent_id"]
            if parent_id == -1:
                continue
            if parent_id in body_hierarchy:
                parent_info = body_hierarchy[parent_id]
                parent_object: SimObject = parent_info["sim_object"]
                parent_object.children.append(body_info["sim_object"])

        return sim_scene

    def process_body(self, obj_id, obj_info):

        # Create SimObject
        body_name = str(obj_id)
        sim_object = SimObject(name=body_name)

        # Check parent
        parent_id = obj_info["parent_id"]
        if parent_id == -1:
            self.sim_scene.root = sim_object

        # Get Transform info, absolute to world
        trans = sim_object.trans
        trans.pos = mj2unity_pos(obj_info["pos"])
        trans.rot = mj2unity_quat(obj_info["quat"])

        # Geometry info
        obj_type = obj_info["type"]
        obj_visualize = obj_info["visualize"]
        if obj_type != "shape":
            # If not a shape, return the object
            return sim_object
        if not obj_visualize: # FIXME, DOUBLE CHECK THIS IF STATEMENT
            return sim_object
        else:
            # Process Material / Texture, fixme 3dim or 4dim?
            ambient_diffuse = obj_info["color_ambient_diffuse"]
            diffuse = obj_info["color_diffuse"]
            specular = obj_info["color_specular"]
            emission = obj_info["color_emission"] # todo, not used
            transparency = obj_info["color_transparency"]
            auxiliary = obj_info["color_auxiliary"]

            # FIXME, THESE COLOR PROPERTIES MISMATCH BETWEEN VREP AND UNITY
            # Fixme, Setting transparency and emission are buggy
            material = SimMaterial(color=ambient_diffuse,
                                   emissionColor=[0., 0., 0., 0.],
                                   # speculPar=0,
                                   # specular=specular,
                                   # shininess=mat_shininess,
                                   # reflectance=mat_reflectance,
                                   # texture=mat_texture
                                   )


            # Process Mesh
            vertices = obj_info["shape_vertices"]
            indices = obj_info["shape_indices"]
            normals = obj_info["shape_normals"]  # fixme, this is not being used
            sim_trans = SimTransform()  # todo, specify some arguments here?
            sim_visual = SimVisual(
                name=body_name,
                type=VisualType.MESH,
                trans=sim_trans,
                # material=SimMaterial(color=[0.5, 0.5, 0.5]),
                material=material,
            )
            sim_visual.mesh = SimMesh.create_mesh(
                scene=self.sim_scene,
                vertices=vertices,
                faces=indices,
                # vertex_normals=normals,
                # mesh_texcoord=mesh_texcoord, # Fixme, texture details?
                # faces_uv=faces_uv,
            )

            sim_object.visuals.append(sim_visual)
            return sim_object

