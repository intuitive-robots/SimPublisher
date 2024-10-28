import mujoco
import numpy as np

from ..simdata import SimObject, SimScene, SimTransform, SimVisual
from .mjcf.utils import str2list, str2listabs, ros2unity, TypeMap

MJModelGeomTypeMap = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
}


class MjModelParser:
    def __init__(self, mj_model):
        self.sim_scene = self.parse_model(mj_model)

    def parse(self):
        return self.sim_scene

    def parse_model(self, mj_model):
        sim_scene = SimScene()
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
                print("Root body found")
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
                trans.pos = ros2unity(mj_model.body_pos[body_id].tolist())
                trans.rot = ros2unity(mj_model.body_quat[body_id].tolist())
        # build the geom information
        for geom_id in range(mj_model.ngeom):
            contype = mj_model.geom_contype[geom_id]
            conaffinity = mj_model.geom_conaffinity[geom_id]
            geom_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if contype == 0 and conaffinity == 0:
                print(f"Geom '{geom_name}'(id {geom_id}) does not participate in collision and can be removed.")
                continue
            print(f"Geom '{geom_name}'(id {geom_id}) will be rendered.")
            geom_type = mj_model.geom_type[geom_id]
            geom_pos = mj_model.geom_pos[geom_id]
            geom_quat = mj_model.geom_quat[geom_id]
            geom_scale = mj_model.geom_size[geom_id]
            geom_color = mj_model.geom_rgba[geom_id]
            trans = SimTransform(
                pos=geom_pos, rot=geom_quat, scale=geom_scale
            )
            sim_visual = SimVisual(
                type=TypeMap[MJModelGeomTypeMap[geom_type]],
                trans=trans,
                color=geom_color,
            )
            # attach mesh id if geom type is mesh
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = mj_model.geom_dataid[geom_id]
                sim_visual.mesh = str(mesh_id)
            # attach material id if geom has an associated material
            mat_id = mj_model.geom_matid[geom_id]
            if mat_id != -1:
                sim_visual.material = str(mat_id)
                print(f"Geom '{geom_name}' have an associated material")
            # attach visual information to the corresponding body
            body_id = mj_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if body_name in body_hierarchy:
                body_info = body_hierarchy[body_name]
                sim_object: SimObject = body_info["sim_object"]
                sim_object.visuals.append(sim_visual)
        return sim_scene

    def process_meshes(self, mj_model):
        # build mesh information
        for mesh_id in range(mj_model.nmesh):
            mesh_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )
            print(f"Mesh: {mesh_name}")
            # get the vertices
            start_vert = mj_model.mesh_vertadr[mesh_id]
            num_verts = mj_model.mesh_vertnum[mesh_id]
            vertices = mj_model.mesh_vert[start_vert:start_vert + num_verts * 3]
            vertices = np.reshape(vertices, (num_verts, 3))
            # convert the vertices
            start_normal = mj_model.mesh_vertadr[mesh_id]
            num_normals = mj_model.mesh_vertnum[mesh_id]
            normals = mj_model.mesh_normal[start_normal:start_normal + num_normals * 3]
            normals = np.reshape(normals, (num_normals, 3))
            print("  Normals:", normals)

            # get the faces
            start_face = mj_model.mesh_faceadr[mesh_id]
            num_faces = mj_model.mesh_facenum[mesh_id]
            faces = mj_model.mesh_face[start_face:start_face + num_faces * 3]
            faces = np.reshape(faces, (num_faces, 3))
            print("  Faces:", faces)

            # get the UV coordinates if available
            if mj_model.mesh_texcoord is not None:
                start_uv = start_vert * 2
                uvs = mj_model.mesh_texcoord[start_uv:start_uv + num_verts * 2]
                uvs = np.reshape(uvs, (num_verts, 2))
                print("  UVs:", uvs)
            else:
                print("  No UV coordinates available for this mesh.")