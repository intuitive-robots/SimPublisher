import mujoco
import numpy as np
import io
from hashlib import md5
from typing import List
from PIL import Image

from ..simdata import SimObject, SimScene, SimTransform, SimVisual, SimMesh
from ..simdata import SimMaterial, SimTexture
from ..core.log import logger
from .mjcf.utils import scale2unity, TypeMap

MJModelGeomTypeMap = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
}


def mj2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat: List[float]) -> List[float]:
    return [quat[2], -quat[3], -quat[1], quat[0]]


class MjModelParser:
    def __init__(self, mj_model):
        self.parse_model(mj_model)

    def parse(self):
        return self.sim_scene

    def parse_model(self, mj_model):
        sim_scene = SimScene()
        self.sim_scene = sim_scene
        # create a dictionary to store the body hierarchy
        body_hierarchy = {}
        for body_id in range(mj_model.nbody):
            body_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            parent_id = mj_model.body_parentid[body_id]
            print(f"Body name: {body_name}, id: {body_id}, parent_id: {parent_id}")
            sim_object = SimObject(name=body_name)
            if parent_id == body_id:
                sim_scene.root = sim_object
                parent_name = "None"
            else:
                parent_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id
                )
            # update the body hierarchy dictionary
            body_hierarchy[body_name] = {
                "body_id": body_id,
                "parent_name": parent_name,
                "sim_object": sim_object,
            }
        # create a tree structure from the body hierarchy
        for body_name, body_info in body_hierarchy.items():
            parent_name = body_info["parent_name"]
            if parent_name == "None":
                continue
            if parent_name in body_hierarchy:
                parent_info = body_hierarchy[parent_name]
                parent_object: SimObject = parent_info["sim_object"]
                parent_object.children.append(body_info["sim_object"])
                body_id = body_info["body_id"]
                sim_object: SimObject = body_info["sim_object"]
                trans = sim_object.trans
                trans.pos = mj2unity_pos(mj_model.body_pos[body_id].tolist())
                trans.rot = mj2unity_quat(mj_model.body_quat[body_id].tolist())
        # build the geom information
        for geom_id in range(mj_model.ngeom):
            geom_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            geom_color = mj_model.geom_rgba[geom_id].tolist()
            # remove the geom if it does not participate in rendering
            if geom_color[3] == 0:
                logger.info(
                    (
                        f"Geom '{geom_name}'(id {geom_id}) does not"
                        f"participate in rendering and can be removed."
                    )
                )
                continue
            geom_type = mj_model.geom_type[geom_id]
            visual_type = TypeMap[MJModelGeomTypeMap[geom_type]]
            geom_pos = mj2unity_pos(mj_model.geom_pos[geom_id].tolist())
            geom_quat = mj2unity_quat(mj_model.geom_quat[geom_id].tolist())
            geom_scale = scale2unity(
                mj_model.geom_size[geom_id].tolist(), visual_type
            )
            trans = SimTransform(
                pos=geom_pos, rot=geom_quat, scale=geom_scale
            )
            sim_visual = SimVisual(
                type=visual_type,
                trans=trans,
                color=geom_color,
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
                mat_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id
                )
                sim_visual.material = mat_name
            # attach visual information to the corresponding body
            body_id = mj_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if body_name in body_hierarchy:
                body_info = body_hierarchy[body_name]
                sim_object: SimObject = body_info["sim_object"]
                sim_object.visuals.append(sim_visual)
        self.process_meshes(mj_model)
        self.process_materials(mj_model)
        self.process_textures(mj_model)
        return sim_scene

    def process_meshes(self, mj_model):
        # build mesh information
        for mesh_id in range(mj_model.nmesh):
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
            start_norm = mj_model.mesh_normaladr[mesh_id]
            num_norm = mj_model.mesh_normalnum[mesh_id]
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
                uvs = np.copy(mj_model.mesh_texcoord[
                    start_uv:start_uv + num_texcoord
                ])
                uvs[:, 1] = 1 - uvs[:, 1]
                uvs = uvs.flatten()
                uv_layout = bin_buffer.tell(), uvs.shape[0]
                bin_buffer.write(uvs)
            # create a SiMmesh object and other data
            bin_data = bin_buffer.getvalue()
            hash = md5(bin_data).hexdigest()
            mesh = SimMesh(
                name=mesh_name,
                indicesLayout=indices_layout,
                verticesLayout=vertices_layout,
                normalsLayout=normal_layout,
                uvLayout=uv_layout,
                dataHash=hash
            )
            self.sim_scene.meshes.append(mesh)
            self.sim_scene.raw_data[mesh.dataHash] = bin_data
            # # get the UV coordinates if available
            # if mj_model.mesh_texcoord is not None:
            #     start_uv = start_vert * 2
            #     uvs = mj_model.mesh_texcoord[start_uv:start_uv + num_verts]
            #     uvs = np.reshape(uvs, (num_verts, 2))
            #     print("  UVs:", uvs)
            # else:
            #     print("  No UV coordinates available for this mesh.")

    def process_materials(self, mj_model):
        # build material information
        for mat_id in range(mj_model.nmat):
            mat_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id
            )
            # mat_id = mat_name,
            mat_color = mj_model.mat_rgba[mat_id].tolist()
            mat_emissionColor = mj_model.mat_emission[mat_id] * np.ones(4)
            mat_emissionColor = mat_emissionColor.tolist()
            mat_specular = float(mj_model.mat_specular[mat_id])
            mat_shininess = float(mj_model.mat_shininess[mat_id])
            mat_reflectance = float(mj_model.mat_reflectance[mat_id])
            tex_id = mj_model.mat_texid[mat_id]
            tex_name = None
            tex_size = (-1, -1)
            # support the 2.x version of mujoco
            if isinstance(tex_id, int):
                if tex_id != -1:
                    tex_name = mujoco.mj_id2name(
                        mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, int(tex_id)
                    )
            # only for mjTEXROLE_RGB which support 3.x version of mujoco
            elif isinstance(tex_id, np.ndarray):
                if tex_id[1] != -1:
                    tex_name = mujoco.mj_id2name(
                        mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id[1]
                    )
            else:
                logger.warning(f"Texture id is of type {type(tex_id)}")
            sim_material = SimMaterial(
                name=mat_name,
                color=mat_color,
                emissionColor=mat_emissionColor,
                specular=mat_specular,
                shininess=mat_shininess,
                reflectance=mat_reflectance,
                texture=tex_name,
                textureSize=tex_size,
            )
            self.sim_scene.materials.append(sim_material)

    def process_textures(self, mj_model):
        # build texture information
        for tex_id in range(mj_model.ntex):
            tex_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id
            )
            if tex_name is None:
                continue
            # assert tex_name is not None, "Texture name is None."
            # tex_type = mj_model.tex_type[tex_id]
            tex_width = mj_model.tex_width[tex_id]
            tex_height = mj_model.tex_height[tex_id]
            tex_data = mj_model.tex_data[tex_id]
            # get the texture data
            start_tex = mj_model.tex_adr[tex_id]
            tex_height = mj_model.tex_height[tex_id]
            tex_width = mj_model.tex_width[tex_id]
            # only we only supported texture channel number is 3
            tex_nchannel = mj_model.tex_nchannel[tex_id]
            assert tex_nchannel == 3, "Only support texture with 3 channels."
            num_tex_data = tex_height * tex_width * tex_nchannel
            tex_data = mj_model.tex_data[start_tex:start_tex + num_tex_data]
            tex_data = np.reshape(
                tex_data, (tex_height, tex_width, tex_nchannel)
            )
            bin_data = Image.fromarray(tex_data, 'RGB').tobytes()
            texture_hash = md5(bin_data).hexdigest()
            texture = SimTexture(
                name=tex_name,
                width=int(tex_width),
                height=int(tex_height),
                # Only support 2d texture
                textureType="2d",
                dataHash=texture_hash
            )
            self.sim_scene.textures.append(texture)
            self.sim_scene.raw_data[texture_hash] = bin_data
