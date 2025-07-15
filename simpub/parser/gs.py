from typing import Tuple, List, Union, Dict
import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink, RigidGeom
import numpy as np


from .simdata import (
    SimScene,
    SimObject,
    SimTransform,
    SimVisual,
    VisualType,
    SimMaterial,
    SimMesh,
)


def gs2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def gs2unity_quat(quat: List[float]) -> List[float]:
    return [quat[2], -quat[3], -quat[1], quat[0]]


class GenesisSceneParser:
    def __init__(self, gs_scene: gs.Scene):
        self.gs_scene = gs_scene
        self.sim_scene, self.update_dict = self.parse_gs_scene(
            gs_scene, ["plane_baselink"]
        )

    def parse_gs_scene(
        self, gs_scene: gs.Scene, no_rendered_objects
    ) -> Tuple[SimScene, Dict[str, Union[RigidEntity, RigidLink]]]:
        sim_scene = SimScene()
        update_dict: Dict[str, Union[RigidEntity, RigidLink]] = {}
        body_hierarchy = {}
        sim_scene.root = SimObject(name="genesis_scene")
        for gs_entity in gs_scene.entities:
            sim_object, idx = self.process_rigid_entity(gs_entity, sim_scene)
            body_hierarchy[idx] = {
                "parent_id": -1,
                "sim_object": sim_object,
            }
            update_dict[sim_object.name] = gs_entity
            for link in gs_entity.links:
                sim_object, parent_id, idx = self.process_link(link, sim_scene)
                # update the body hierarchy dictionary
                body_hierarchy[idx] = {
                    "parent_id": parent_id,
                    "sim_object": sim_object,
                }
                update_dict[sim_object.name] = link
        for body_info in body_hierarchy.values():
            parent_id = body_info["parent_id"]
            if parent_id == -1:
                sim_scene.root.children.append(body_info["sim_object"])
            if parent_id in body_hierarchy:
                parent_info = body_hierarchy[parent_id]
                parent_object: SimObject = parent_info["sim_object"]
                parent_object.children.append(body_info["sim_object"])
        return sim_scene, update_dict

    def process_rigid_entity(
        self, gs_rigid_entity: RigidEntity, sim_scene: SimScene
    ) -> Tuple[SimObject, int]:
        sim_object = SimObject(name=str(gs_rigid_entity.uid))
        pos: List[float] = [0.0, 0.0, 0.0]
        rot: List[float] = [0.0, 0.0, 0.0, 1.0]
        if self.gs_scene.is_built:
            pos = gs_rigid_entity.get_pos()
            rot = gs_rigid_entity.get_quat()
            assert type(pos) is List[float]
            assert type(rot) is List[float]
        sim_object.trans = SimTransform(
            gs2unity_pos(pos),
            gs2unity_quat(rot),
        )
        return sim_object, gs_rigid_entity.base_link_idx

    def process_link(
        self, gs_link: RigidLink, sim_scene: SimScene
    ) -> Tuple[SimObject, int, int]:
        sim_object = SimObject(name=gs_link.name + f"{np.random.rand()}")
        sim_object.trans = SimTransform(
            gs2unity_pos(gs_link.pos),
            gs2unity_quat(gs_link.quat),
        )
        if gs_link.name in ["plane_baselink"]:
            return sim_object, gs_link.parent_idx, gs_link.idx
        for gs_geom in gs_link.vgeoms:
            sim_visual = self.process_vgeom(gs_geom, sim_scene)
            sim_object.visuals.append(sim_visual)
        return sim_object, gs_link.parent_idx, gs_link.idx

    def process_vgeom(
        self, gs_vgeom: RigidGeom, sim_scene: SimScene
    ) -> SimVisual:
        sim_visual = SimVisual(
            str(gs_vgeom.uid),
            type=VisualType.MESH,
            trans=SimTransform(
                gs2unity_pos(gs_vgeom.init_pos),
                gs2unity_quat(gs_vgeom.init_quat),
            ),
        )
        sim_visual.mesh = self.process_mesh(gs_vgeom, sim_scene)
        # sim_visual.material = SimMaterial(
        #     color=[1.0, 1.0, 1.0, 1.0],
        # )
        sim_visual.material = self.parse_material(gs_vgeom)
        return sim_visual

    def process_mesh(
        self, gs_vgeom: RigidGeom, sim_scene: SimScene
    ) -> SimMesh:
        gs_trimesh_obj = gs_vgeom.get_trimesh()
        return SimMesh.create_mesh(
            sim_scene,
            gs_trimesh_obj.vertices,
            gs_trimesh_obj.faces,
        )

    # TODO: Implement the material and texture from trimesh
    def parse_material(self, gs_vgeom: RigidGeom):
        gs_trimesh_obj = gs_vgeom.get_trimesh()
        if isinstance(
            gs_trimesh_obj.visual, gs.ext.trimesh.visual.color.ColorVisuals
        ):
            return SimMaterial(
                color=list(gs_trimesh_obj.visual.face_colors[0] / 255.0)
            )
        else:
            return SimMaterial(color=[0.0, 0.0, 0.0, 1.0])

    # def parse_mesh(self, gs_mesh: gs.Mesh):
    #     pass
