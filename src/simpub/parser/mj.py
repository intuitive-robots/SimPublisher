from typing import Callable, Dict, List, Tuple

import mujoco
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

from ..core.log import logger
from .simdata import (
    SimMaterial,
    TreeNode,
    create_mesh,
    create_texture,
    create_material,
    SimObject,
    SimScene,
    LightConfig,
    SimTransform,
    SimVisual,
    VisualType,
    SimSceneConfig
)


def plane2unity_scale(scale: List[float]) -> List[float]:
    return list(map(abs, [scale[0] * 2, 0.001, scale[1] * 2]))


def box2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[i]) * 2 for i in [1, 2, 0]]


def sphere2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[0]) * 2] * 3


def cylinder2unity_scale(scale: List[float]) -> List[float]:
    if len(scale) == 3:
        return list(map(abs, [scale[0] * 2, scale[1], scale[0] * 2]))
    if len(scale) == 2:
        return list(map(abs, [scale[0] * 2, scale[1], scale[0] * 2]))
    elif len(scale) == 1:
        return list(map(abs, [scale[0] * 2, scale[0] * 2, scale[0] * 2]))
    raise ValueError("Only support scale with one, two or three components.")


def capsule2unity_scale(scale: List[float]) -> List[float]:
    if len(scale) == 2:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    elif len(scale) == 1:
        return list(map(abs, [scale[0] * 2, scale[0] * 2, scale[0] * 2]))
    elif len(scale) == 3:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    raise ValueError("Only support scale with one, two or three components.")


ScaleMap: Dict[str, Callable] = {
    VisualType.PLANE: lambda x: plane2unity_scale(x),
    VisualType.CUBE: lambda x: box2unity_scale(x),
    VisualType.SPHERE: lambda x: sphere2unity_scale(x),
    VisualType.CYLINDER: lambda x: cylinder2unity_scale(x),
    VisualType.CAPSULE: lambda x: capsule2unity_scale(x),
    VisualType.NONE: lambda x: capsule2unity_scale(x),
}

def scale2unity(scale: List[float], visual_type: str) -> List[float]:
    if visual_type in ScaleMap:
        return ScaleMap[visual_type](scale)
    else:
        return [1, 1, 1]


MJModelGeomTypeMap = {
    mujoco.mjtGeom.mjGEOM_SPHERE: VisualType.SPHERE,  # type: ignore
    mujoco.mjtGeom.mjGEOM_CAPSULE: VisualType.CAPSULE,  # type: ignore
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: VisualType.CAPSULE,  # type: ignore
    mujoco.mjtGeom.mjGEOM_CYLINDER: VisualType.CYLINDER,  # type: ignore
    mujoco.mjtGeom.mjGEOM_BOX: VisualType.CUBE,  # type: ignore
    mujoco.mjtGeom.mjGEOM_MESH: VisualType.MESH,  # type: ignore
    mujoco.mjtGeom.mjGEOM_PLANE: VisualType.PLANE,  # type: ignore
}


def mj2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat: List[float]) -> List[float]:
    return [quat[2], -quat[3], -quat[1], quat[0]]


class MjModelParser:
    def __init__(
        self, mj_model, visible_geoms_groups, no_rendered_objects=None
    ):
        if no_rendered_objects is None:
            self.no_rendered_objects = []
        else:
            self.no_rendered_objects = no_rendered_objects
        self.visible_geoms_groups = visible_geoms_groups
        self.parse_config(mj_model)
        self.parse_model(mj_model)
        self.sim_scene.process_sim_obj(self.sim_scene.root)

    def parse(self):
        return self.sim_scene

    def parse_config(self, mj_model):
        self.sim_scene = SimScene(
            SimSceneConfig(
                name="MujocoScene",
                pos=[0, 0, 0],
                rot=[0, 0, 0, 1],
                scale=[1, 1, 1],
            ),
        )
        self.sim_scene.lights = self.process_lights(mj_model)

    def parse_model(self, mj_model):
        sim_scene = self.sim_scene
        # create a dictionary to store the body hierarchy
        body_hierarchy: Dict[int, Tuple[int, TreeNode]] = {}
        for body_id in range(mj_model.nbody):
            object_node, parent_id = self.process_body(mj_model, body_id)
            # if object_node is None:
            #     continue
            # update the body hierarchy dictionary
            body_hierarchy[body_id] = (parent_id, object_node)
        # create a tree structure from the body hierarchy
        for body_id, body_info in body_hierarchy.items():
            parent_id = body_info[0]
            if parent_id == -1:
                sim_scene.root = body_info[1]
            if parent_id in body_hierarchy:
                parent_info = body_hierarchy[parent_id]
                parent_object: TreeNode = parent_info[1]
                # assert parent_object.data is not None, "Parent object data should not be None."
                parent_object.children.append(body_info[1])
        assert sim_scene.root is not None, "The root of the SimScene is None."
        return sim_scene

    def process_body(self, mj_model, body_id: int) -> Tuple[TreeNode, int]:
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)  # type: ignore
        # if body_name in self.no_rendered_objects:
        #     return None, -1
        parent_id = mj_model.body_parentid[body_id]
        parent_name = "root"
        if parent_id != -1:
            parent_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)  # type: ignore
        object_node = TreeNode()
        if parent_id == body_id:
            parent_id = -1
        # update the transform information
        trans = SimTransform(
            pos=mj2unity_pos(mj_model.body_pos[body_id].tolist()),
            rot=mj2unity_quat(mj_model.body_quat[body_id].tolist()),
            scale=[1, 1, 1],
        )
        visuals_list = []
        if (
            mj_model.body_geomadr[body_id] != -1
            and body_name not in self.no_rendered_objects
        ):
            # process the geoms attached to the body
            num_geoms = mj_model.body_geomnum[body_id]
            for geom_id in range(
                mj_model.body_geomadr[body_id],
                mj_model.body_geomadr[body_id] + num_geoms,
            ):
                geom_group = int(mj_model.geom_group[geom_id])
                # check if the geom participates in rendering
                if geom_group not in self.visible_geoms_groups:
                    continue
                sim_visual = self.process_geoms(mj_model, geom_id)
                if sim_visual["name"] is None:
                    sim_visual["name"] = f"{body_name}_geom_{geom_id}"
                visuals_list.append(sim_visual)
        object_node.data = SimObject(name=body_name, parent=parent_name, trans=trans, visuals=visuals_list)
        assert object_node.data is not None, "Object node data should not be None."
        return object_node, parent_id

    def process_geoms(self, mj_model, geom_id: int):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)  # type: ignore
        geom_type = mj_model.geom_type[geom_id]
        geom_pos = mj2unity_pos(mj_model.geom_pos[geom_id].tolist())
        geom_quat = mj2unity_quat(mj_model.geom_quat[geom_id].tolist())
        visual_type = MJModelGeomTypeMap[geom_type]
        geom_scale = scale2unity(
            mj_model.geom_size[geom_id].tolist(), visual_type
        )
        trans = SimTransform(pos=geom_pos, rot=geom_quat, scale=geom_scale)
        geom_color = mj_model.geom_rgba[geom_id].tolist()
        # attach mesh id if geom type is mesh
        mesh = None
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:  # type: ignore
            mesh_id = mj_model.geom_dataid[geom_id]
            mesh = self.process_mesh(mj_model, mesh_id)
        # attach material id if geom has an associated material
        mat_id = mj_model.geom_matid[geom_id]
        material = create_material(geom_color)
        if mat_id != -1:
            material = self.process_material(mj_model, mat_id)
        return SimVisual(
            name=geom_name,
            type=visual_type,
            mesh=mesh,
            material=material,
            trans=trans,
        )

    def process_mesh(self, mj_model, mesh_id: int):
        # vertices
        start_vert = mj_model.mesh_vertadr[mesh_id]
        num_verts = mj_model.mesh_vertnum[mesh_id]
        vertices = mj_model.mesh_vert[start_vert : start_vert + num_verts]
        vertices = vertices.astype(np.float32)
        # faces
        start_face = mj_model.mesh_faceadr[mesh_id]
        num_faces = mj_model.mesh_facenum[mesh_id]
        faces = mj_model.mesh_face[start_face : start_face + num_faces]
        faces = faces.astype(np.int32)
        # uv
        start_uv = mj_model.mesh_texcoordadr[mesh_id]
        faces_uv = None
        if start_uv != -1:
            num_texcoord = mj_model.mesh_texcoordnum[mesh_id]
            all_uv_coords = mj_model.mesh_texcoord[start_uv : start_uv + num_texcoord]
            faces_uv_idx = mj_model.mesh_facetexcoord[start_face : start_face + num_faces]
            flattened_uvs = all_uv_coords[faces_uv_idx].reshape(-1, 2)
            trimesh_obj = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=TextureVisuals(uv=flattened_uvs),
                process=False,
            )
        else:
            trimesh_obj = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )
        trimesh_obj.unmerge_vertices()
        trimesh_obj.fix_normals()
        # faces_uv = faces_uv.astype(np.int32) if faces_uv is not None else None
        return create_mesh(trimesh_obj, faces_uv)

    def process_material(self, mj_model, mat_id: int):
        # build material information
        mat_color = mj_model.mat_rgba[mat_id]
        mat_color = mat_color.tolist()
        # TODO: emission color processing
        mat_emissionColor = [0.0, 0.0, 0.0, 1.0]
        # mat_emissionColor = mj_model.mat_emission[mat_id] * mat_color
        # mat_emissionColor = mat_emissionColor.tolist()
        mat_specular = float(mj_model.mat_specular[mat_id])
        mat_shininess = float(mj_model.mat_shininess[mat_id])
        mat_reflectance = float(mj_model.mat_reflectance[mat_id])
        tex_id = mj_model.mat_texid[mat_id]
        mat_texture = None
        # support the 2.x version of mujoco
        if isinstance(tex_id, np.integer):
            tex_id = int(tex_id)
        # only for mjTEXROLE_RGB which support 3.x version of mujoco
        # The second element of the texture array is base color (albedo)
        # TODO： Check after which mujoco version the tex id array is supported
        elif isinstance(tex_id, np.ndarray):
            tex_id = int(tex_id[1])
        else:
            logger.warning(
                f"Texture id is of type {type(tex_id)},"
                "which is not supported."
            )
        if tex_id != -1:
            mat_texture = self.process_texture(mj_model, tex_id)
            mat_texture["textureScale"] = mj_model.mat_texrepeat[mat_id].tolist()
        material = SimMaterial(
            color=mat_color,
            emissionColor=mat_emissionColor,
            specular=mat_specular,
            shininess=mat_shininess,
            reflectance=mat_reflectance,
            texture=mat_texture,
        )
        return material

    def process_texture(self, mj_model, tex_id: int):
        # build texture information
        # TODO: Support for more texture types
        # get the texture data
        tex_height = mj_model.tex_height[tex_id].item()
        tex_width = mj_model.tex_width[tex_id].item()
        # only we only supported texture channel number is 3
        if hasattr(mj_model, "tex_nchannel"):
            tex_nchannel = mj_model.tex_nchannel[tex_id]
        else:
            tex_nchannel = 3
        assert tex_nchannel == 3, "Only support texture with 3 channels."
        start_tex = mj_model.tex_adr[tex_id]
        num_tex_data = tex_height * tex_width * tex_nchannel
        if hasattr(mj_model, "tex_data"):
            tex_data: np.ndarray = mj_model.tex_data[
                start_tex : start_tex + num_tex_data
            ]
        else:
            tex_data: np.ndarray = mj_model.tex_rgb[
                start_tex : start_tex + num_tex_data
            ]
        return create_texture(tex_data, tex_height, tex_width)

    def process_lights(self, mj_model) -> List[LightConfig]:
        """
        Extracts light data from MuJoCo mjModel and mjData and converts
        them to Unity-compatible coordinates and formats.
        """
        lights = []
        
        for i in range(mj_model.nlight):
            # 1. Get Light Name from MuJoCo constants
            name_adr = mj_model.name_lightadr[i]
            light_name = mj_model.names[name_adr:].split(b'\x00')[0].decode()
            parent_id = mj_model.light_bodyid[i]
            parent_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)  # type: ignore
            if parent_name is None:
                parent_name = "root"
            is_directional = bool(mj_model.light_directional[i])
            # If cutoff is small, it acts as a Spot light; if 180, it's a Point light
            cutoff = float(mj_model.light_cutoff[i])
            if is_directional:
                l_type = "Directional"
            elif cutoff < 180:
                l_type = "Spot"
            else:
                l_type = "Point"

            # 4. Map MuJoCo properties to Unity Light fields
            config: LightConfig = {
                "name": light_name,
                "parent": parent_name,
                "lightType": l_type,
                "color": mj_model.light_diffuse[i].tolist(),
                "intensity": 1.0,  # Default scale, can be tuned based on 'light_attenuation'
                "position": mj2unity_pos(mj_model.light_pos[i].tolist()),
                "direction": mj2unity_pos(mj_model.light_dir[i].tolist()),
                "range": 20.0,     # Unity specific: distance where light hits zero
                "spotAngle": cutoff * 2, # MuJoCo cutoff is half-angle, Unity is full-angle
                "shadowType": "Soft" if mj_model.light_castshadow[i] else "None"
            }
            lights.append(config)
        return lights