import mujoco
import numpy as np
import io
from hashlib import md5
from typing import List, Dict, Callable

from ..simdata import SimObject, SimScene, SimTransform, SimVisual, SimMesh
from ..simdata import SimMaterial, SimTexture
from ..simdata import VisualType
from ..core.log import logger


def plane2unity_scale(scale: List[float]) -> List[float]:
    return list(map(abs, [scale[0] * 2, 0.001, scale[1] * 2]))


def box2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[i]) * 2 for i in [1, 2, 0]]


def sphere2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[0]) * 2] * 3


def cylinder2unity_scale(scale: List[float]) -> List[float]:
    if len(scale) == 3:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    if len(scale) == 2:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    elif len(scale) == 1:
        return list(map(abs, [scale[0] * 2, scale[0] * 2, scale[0] * 2]))


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
    mujoco.mjtGeom.mjGEOM_SPHERE: VisualType.SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE: VisualType.CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: VisualType.CAPSULE,
    mujoco.mjtGeom.mjGEOM_CYLINDER: VisualType.CYLINDER,
    mujoco.mjtGeom.mjGEOM_BOX: VisualType.CUBE,
    mujoco.mjtGeom.mjGEOM_MESH: VisualType.MESH,
    mujoco.mjtGeom.mjGEOM_PLANE: VisualType.PLANE,
}


def mj2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat: List[float]) -> List[float]:
    return [quat[2], -quat[3], -quat[1], quat[0]]


class MjModelParser:
    def __init__(self, mj_model, visible_geoms_groups):
        self.visible_geoms_groups = visible_geoms_groups
        self.parse_model(mj_model)

    def parse(self):
        return self.sim_scene

    def parse_model(self, mj_model):
        sim_scene = SimScene()
        self.sim_scene = sim_scene
        # create a dictionary to store the body hierarchy
        body_hierarchy = {}
        for body_id in range(mj_model.nbody):
            sim_object, parent_id = self.process_body(
                mj_model, body_id
            )
            # update the body hierarchy dictionary
            body_hierarchy[body_id] = {
                "parent_id": parent_id,
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

    def process_body(self, mj_model, body_id: int):
        body_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        parent_id = mj_model.body_parentid[body_id]
        sim_object = SimObject(name=body_name)
        if parent_id == body_id:
            self.sim_scene.root = sim_object
            parent_id = -1
        # update the transform information
        trans = sim_object.trans
        trans.pos = mj2unity_pos(mj_model.body_pos[body_id].tolist())
        trans.rot = mj2unity_quat(mj_model.body_quat[body_id].tolist())
        # if the body does not have any geom, return the object
        if mj_model.body_geomadr[body_id] == -1:
            return sim_object, parent_id
        # process the geoms attached to the body
        num_geoms = mj_model.body_geomnum[body_id]
        for geom_id in range(
            mj_model.body_geomadr[body_id],
            mj_model.body_geomadr[body_id + num_geoms]
        ):
            geom_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            geom_group = int(mj_model.geom_group[geom_id])
            # check if the geom participates in rendering
            if geom_group not in self.visible_geoms_groups:
                logger.info(
                    (
                        f"Geom '{geom_name}'(id {geom_id}) does not"
                        f"participate in rendering and can be removed."
                    )
                )
                continue
            sim_visual = self.process_geoms(mj_model, geom_id)
            sim_object.visuals.append(sim_visual)
        return sim_object, parent_id

    def process_geoms(self, mj_model, geom_id: int):
        geom_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        geom_type = mj_model.geom_type[geom_id]
        geom_pos = mj2unity_pos(mj_model.geom_pos[geom_id].tolist())
        geom_quat = mj2unity_quat(mj_model.geom_quat[geom_id].tolist())
        visual_type = MJModelGeomTypeMap[geom_type]
        geom_scale = scale2unity(
            mj_model.geom_size[geom_id].tolist(), visual_type
        )
        trans = SimTransform(
            pos=geom_pos, rot=geom_quat, scale=geom_scale
        )
        geom_color = mj_model.geom_rgba[geom_id].tolist()
        sim_visual = SimVisual(
            name=geom_name,
            type=visual_type,
            trans=trans,
        )
        # attach mesh id if geom type is mesh
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = mj_model.geom_dataid[geom_id]
            mesh_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )
            sim_visual.mesh = mesh_name
        # attach material id if geom has an associated material
        mat_id = mj_model.geom_matid[geom_id]
        if mat_id != -1:
            sim_visual.material = self.process_material(
                mj_model, mat_id
            )
        else:
            sim_visual.material = SimMaterial()
            sim_visual.material.color = geom_color
        return sim_visual

    def process_mesh(self, mj_model, mesh_id: int):
        # build mesh information
        mesh_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
        )
        bin_buffer = io.BytesIO()
        # vertices
        start_vert = mj_model.mesh_vertadr[mesh_id]
        num_verts = mj_model.mesh_vertnum[mesh_id]
        vertices = mj_model.mesh_vert[start_vert:start_vert + num_verts]
        vertices = vertices.astype(np.float32)
        vertices = vertices[:, [1, 2, 0]]
        vertices[:, 0] = - vertices[:, 0]
        vertices = vertices.flatten()
        vertices_layout = bin_buffer.tell(), vertices.shape[0]
        bin_buffer.write(vertices)
        # normal
        if hasattr(mj_model, "mesh_normaladr"): 
            start_norm = mj_model.mesh_normaladr[mesh_id]
            num_norm = mj_model.mesh_normalnum[mesh_id]
        else:
            start_norm = start_vert
            num_norm = num_verts
        norms = mj_model.mesh_normal[start_norm:start_norm + num_norm]
        norms = norms.astype(np.float32)
        norms = norms[:, [1, 2, 0]]
        norms[:, 0] = - norms[:, 0]
        norms = norms.flatten()
        normal_layout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms)
        # faces
        start_face = mj_model.mesh_faceadr[mesh_id]
        num_faces = mj_model.mesh_facenum[mesh_id]
        faces = mj_model.mesh_face[start_face:start_face + num_faces]
        indices = faces.astype(np.int32)
        indices = indices[:, [2, 1, 0]]
        indices = indices.flatten()
        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)
        # Texture coords
        uv_layout = (0, 0)
        start_uv = mj_model.mesh_texcoordadr[mesh_id]
        if start_uv != -1:
            num_texcoord = mj_model.mesh_texcoordnum[mesh_id]
            if num_texcoord > num_verts:
                num_texcoord = num_verts
            # TODO: fill in the missing texture coordinates
            # TODO: YCB object in SimulationFramework is not worked
            uvs = np.copy(mj_model.mesh_texcoord[
                start_uv:start_uv + num_texcoord
            ])
            uvs = 1 - uvs
            uvs = uvs.flatten()
            uv_layout = bin_buffer.tell(), uvs.shape[0]
            bin_buffer.write(uvs)
        # create a SiMmesh object and raw data
        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()
        mesh = SimMesh(
            name=mesh_name,
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            normalsLayout=normal_layout,
            uvLayout=uv_layout,
            hash=hash
        )
        self.sim_scene.raw_data[mesh.hash] = bin_data
        return mesh

    def process_material(self, mj_model, mat_id: int):
        # build material information
        mat_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id
        )
        # mat_id = mat_name,
        mat_color = mj_model.mat_rgba[mat_id]
        mat_emissionColor = mj_model.mat_emission[mat_id] * mat_color
        mat_color = mat_color.tolist()
        mat_emissionColor = mat_emissionColor.tolist()
        mat_specular = float(mj_model.mat_specular[mat_id])
        mat_shininess = float(mj_model.mat_shininess[mat_id])
        mat_reflectance = float(mj_model.mat_reflectance[mat_id])
        tex_id = mj_model.mat_texid[mat_id]
        mat_texture = None
        # support the 2.x version of mujoco
        if isinstance(tex_id, np.int32):
            if tex_id != -1:
                tex_id = int(tex_id)
        # only for mjTEXROLE_RGB which support 3.x version of mujoco
        elif isinstance(tex_id, np.ndarray):
            if tex_id[1] != -1:
                tex_id = int(tex_id[1])
        else:
            logger.warning(
                f"Texture id is of type {type(tex_id)},"
                "which is not supported."
            )
        if tex_id == -1:
            mat_texture = self.process_geoms(mj_model, tex_id)
        material = SimMaterial(
            name=mat_name,
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
        tex_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id
        )
        # TODO: Texture type?
        # tex_type = mj_model.tex_type[tex_id]
        # get the texture data
        tex_height = mj_model.tex_height[tex_id]
        tex_width = mj_model.tex_width[tex_id]
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
                start_tex:start_tex + num_tex_data
            ]
        else:
            tex_data: np.ndarray = mj_model.tex_rgb[
                start_tex:start_tex + num_tex_data
            ]
        bin_data = tex_data.tobytes()
        texture_hash = md5(bin_data).hexdigest()
        texture = SimTexture(
            name=tex_name,
            width=int(tex_width),
            height=int(tex_height),
            # Only support 2D texture
            textureType="2D",
            hash=texture_hash
        )
        self.sim_scene.raw_data[texture_hash] = bin_data
        return texture
