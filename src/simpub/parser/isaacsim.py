import io
import json
import math
import os
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyzlc
import requests
import trimesh
import trimesh.visual
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, UsdUtils
from usdrt import Usd as RtUsd
from usdrt import UsdGeom as RtGeom
from importlib.util import find_spec

import omni
import omni.usd

from .simdata import (
    SimMaterial,
    SimObject,
    SimScene,
    SimSceneConfig,
    SimTexture,
    SimTransform,
    SimVisual,
    TreeNode,
    VisualType,
    create_material,
    create_mesh,
    create_texture,
)


# Detect Isaac Sim version to support both old and new package layouts.
def _get_isaacsim_version() -> str:
    try:
        from importlib.metadata import version

        return version("isaacsim")
    except Exception:
        try:
            return "5.0" if find_spec("isaacsim.core.prims") is not None else "4.0"
        except ImportError:
            return "4.0"


_ISAACSIM_VERSION = _get_isaacsim_version()

if _ISAACSIM_VERSION < "4.5":
    from omni.isaac.core.prims import XFormPrim as SingleXFormPrim
    from omni.isaac.core.utils.rotations import (
        euler_angles_to_quat,
        quat_to_rot_matrix,
    )
else:
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.rotations import (
        euler_angles_to_quat,
        quat_to_rot_matrix,
    )

MaterialContext = Tuple[SimMaterial, bool, bool]


class IsaacSimStageParser:
    def __init__(
        self,
        stage: Usd.Stage,
        ignored_prim_paths: Optional[List[str]] = None,
        texture_cache_dir: Optional[str] = None,
    ) -> None:
        assert isinstance(stage, Usd.Stage)
        self.stage = stage
        self.ignored_prim_paths = set(ignored_prim_paths or [])

        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        self.rt_stage = RtUsd.Stage.Attach(stage_id)

        self.tracked_prims: List[dict] = []
        self.tracked_deform_prims: List[dict] = []

        self.texture_cache_dir = texture_cache_dir
        self.texture_dict: Dict[str, Dict[str, Optional[object]]] = {}
        self.texture_dict_path: Optional[str] = None
        self._load_texture_cache()

        self.sim_scene: Optional[SimScene] = None

    def get_usdrt_stage(self) -> RtUsd.Stage:
        return self.rt_stage

    def get_tracked_prims(self) -> Tuple[List[dict], List[dict]]:
        return self.tracked_prims, self.tracked_deform_prims

    def parse_scene(self) -> SimScene:
        self.parse_config()
        self.parse_model()
        assert self.sim_scene is not None
        self.sim_scene.process_sim_obj(self.sim_scene.root)
        self._store_texture_cache()
        return self.sim_scene

    def parse_config(self) -> None:
        self.sim_scene = SimScene(
            SimSceneConfig(
                name="IsaacSimScene",
                pos=[0, 0, 0],
                rot=[0, 0, 0, 1],
                scale=[1, 1, 1],
            ),
        )

    def parse_model(self) -> None:
        assert self.sim_scene is not None
        world_prim = self.stage.GetPrimAtPath("/World")
        root_node = self.parse_prim_tree(world_prim)
        if root_node is None:
            raise RuntimeError("Failed to parse /World prim")
        self.sim_scene.root = root_node

    def parse_prim_tree(
        self,
        root: Usd.Prim,
        indent: int = 0,
        parent_path: Optional[str] = None,
        inherited_material: Optional[MaterialContext] = None,
    ) -> Optional[TreeNode]:
        if not root or not root.IsValid():
            return None

        root_path = str(root.GetPath())
        if root_path in self.ignored_prim_paths:
            return None

        if root.GetTypeName() not in {
            "",
            "Xform",
            "Mesh",
            "Scope",
            "Cube",
            "Capsule",
            "Cone",
            "Cylinder",
            "Sphere",
        }:
            return None

        purpose_attr = root.GetAttribute("purpose")
        if purpose_attr and purpose_attr.Get() in {"proxy", "guide"}:
            return None

        prim_path = (
            root_path if parent_path is None else f"{parent_path}/{root.GetName()}"
        )

        translate, rot, scale = self.compute_local_trans(root)
        sim_object: SimObject = {
            "name": prim_path.replace("/", "_"),
            "parent": "root" if parent_path is None else parent_path.replace("/", "_"),
            "trans": SimTransform(pos=translate, rot=rot, scale=scale),
            "visuals": [],
        }

        active_material = self.parse_prim_geometries(
            prim=root,
            prim_path=prim_path,
            sim_obj=sim_object,
            indent=indent,
            inherited_material=inherited_material,
        )

        self._track_prim_if_needed(root, sim_object["name"], prim_path, indent)

        node = TreeNode()
        node.data = sim_object

        children_src = (
            root.GetPrototype().GetChildren()
            if root.IsInstance()
            else root.GetChildren()
        )
        for child in children_src:
            child_node = self.parse_prim_tree(
                root=child,
                indent=indent + 1,
                parent_path=prim_path,
                inherited_material=active_material,
            )
            if child_node is not None:
                node.children.append(child_node)

        return node

    def compute_local_trans(
        self, prim: Usd.Prim
    ) -> Tuple[List[float], List[float], List[float]]:
        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
        sc, rt, rto, tr = omni.usd.get_local_transform_SRT(prim, timecode)

        scale = [float(sc[1]), float(sc[2]), float(sc[0])]
        translate = [float(tr[1]), float(tr[2]), float(-tr[0])]

        rtq = euler_angles_to_quat([rt[rto[0]], rt[rto[1]], rt[rto[2]]], True)
        rot = [float(-rtq[2]), float(-rtq[3]), float(rtq[1]), float(rtq[0])]
        return translate, rot, scale

    def compute_world_trans(
        self, prim: Usd.Prim
    ) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        xform = SingleXFormPrim(str(prim.GetPath()))
        assert xform.is_valid()

        pos, quat = xform.get_world_pose()
        scale = xform.get_world_scale()

        return (
            pos.cpu().numpy().astype(np.float32),
            quat_to_rot_matrix(quat.cpu().numpy()).astype(np.float32),
            scale.cpu().numpy().astype(np.float32),
        )

    def process_prim_material(
        self,
        prim: Usd.Prim,
        indent: int = 0,
    ) -> Optional[SimMaterial]:
        matapi = UsdShade.MaterialBindingAPI(prim)
        if matapi is None:
            return None

        binding = matapi.GetDirectBinding()
        if not binding:
            return None

        mat = binding.GetMaterial()
        if not mat:
            return None

        mat_prim = self.stage.GetPrimAtPath(mat.GetPath())
        if not mat_prim:
            return None

        shader_children = mat_prim.GetAllChildren()
        if not shader_children:
            return None

        mat_shader = UsdShade.Shader(shader_children[0])
        texture_path = self._resolve_texture_path(mat_shader)
        diffuse_color = self._resolve_diffuse_color(mat_shader)

        sim_mat = create_material(color=diffuse_color + [1.0])

        if texture_path is not None:
            image = self._load_texture_image(texture_path)
            if image is not None:
                image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
                tex: SimTexture = create_texture(
                    image_flaten_array=image_np.reshape(-1, 3),
                    image_height=image_np.shape[0],
                    image_width=image_np.shape[1],
                )
                sim_mat["texture"] = tex

        return sim_mat

    def _resolve_prim_projection_mode(self, prim: Usd.Prim) -> Tuple[bool, bool]:
        matapi = UsdShade.MaterialBindingAPI(prim)
        if matapi is None:
            return False, False

        binding = matapi.GetDirectBinding()
        if not binding:
            return False, False

        mat = binding.GetMaterial()
        if not mat:
            return False, False

        mat_prim = self.stage.GetPrimAtPath(mat.GetPath())
        if not mat_prim:
            return False, False

        shader_children = mat_prim.GetAllChildren()
        if not shader_children:
            return False, False

        mat_shader = UsdShade.Shader(shader_children[0])
        project_uvw_input = mat_shader.GetInput("project_uvw")
        if project_uvw_input and project_uvw_input.Get() is True:
            world_coord_input = mat_shader.GetInput("world_or_object")
            use_world_coord = bool(world_coord_input and world_coord_input.Get())
            return True, use_world_coord

        return False, False

    def compute_projected_uv(
        self,
        prim: Usd.Prim,
        vertex_buf: np.ndarray,
        index_buf: np.ndarray,
        use_world_coord: bool = False,
    ) -> np.ndarray:
        assert len(vertex_buf.shape) == 2 and vertex_buf.shape[1] in {3, 4}
        assert len(index_buf.shape) == 2 and index_buf.shape[1] in {3, 4}

        uvs: List[List[float]] = []
        axes = np.array(
            [
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
            ],
            dtype=np.float32,
        )
        axis_projectors = [
            lambda v: [v[0], v[1]],
            lambda v: [v[0], -v[1]],
            lambda v: [v[0], v[2]],
            lambda v: [-v[0], v[2]],
            lambda v: [v[1], v[2]],
            lambda v: [-v[1], v[2]],
        ]

        for face in index_buf:
            points = [vertex_buf[idx][:3] for idx in face]
            if use_world_coord:
                pos, rot_m, scale = self.compute_world_trans(prim)
                points = [rot_m @ (p * scale) + pos for p in points]

            p0, p1, p2 = points[0], points[1], points[2]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm

            axis_id = int(np.argmax(axes @ normal))
            projector = axis_projectors[axis_id]
            for p in points:
                uvs.append(projector(p))

        return np.asarray(uvs, dtype=np.float32)

    def parse_prim_geometries(
        self,
        prim: Usd.Prim,
        prim_path: str,
        sim_obj: SimObject,
        indent: int,
        inherited_material: Optional[MaterialContext] = None,
    ) -> Optional[MaterialContext]:
        visibility_attr = prim.GetAttribute("visibility")
        if visibility_attr and str(visibility_attr.Get()) == "invisible":
            return inherited_material

        resolved_material = self.process_prim_material(prim, indent=indent)
        if resolved_material is None:
            active_material = inherited_material
        else:
            project_uvw, use_world_coord = self._resolve_prim_projection_mode(prim)
            active_material = (resolved_material, project_uvw, use_world_coord)

        prim_type = prim.GetTypeName()
        if prim_type == "Mesh":
            self._process_mesh_prim(prim, sim_obj, indent, active_material)
            return active_material

        primitive_visual = self._process_primitive_prim(
            prim_type, prim_path, active_material
        )
        if primitive_visual is not None:
            sim_obj["visuals"].append(primitive_visual)

        return active_material

    def build_mesh_buffer(self, mesh_obj: trimesh.Trimesh) -> SimVisual:
        mesh_data = create_mesh(mesh_obj, None)
        return SimVisual(
            name=str(uuid.uuid4()),
            type=VisualType.MESH,
            mesh=mesh_data,
            material=create_material(color=[1.0, 1.0, 1.0, 1.0]),
            trans=SimTransform(pos=[0, 0, 0], rot=[0, 0, 0, 1], scale=[1, 1, 1]),
        )

    def _load_texture_cache(self) -> None:
        if self.texture_cache_dir is None:
            return

        if not os.path.isdir(self.texture_cache_dir):
            os.makedirs(self.texture_cache_dir, exist_ok=True)

        self.texture_dict_path = os.path.join(
            self.texture_cache_dir, "texture_dict.json"
        )
        if not os.path.isfile(self.texture_dict_path):
            return

        with open(self.texture_dict_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)

        for full_path, relative_path in raw_dict.items():
            self.texture_dict[full_path] = {
                "relative_path": relative_path,
                "image": None,
            }

    def _store_texture_cache(self) -> None:
        if self.texture_dict_path is None:
            return

        with open(self.texture_dict_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: str(v["relative_path"])
                    for k, v in self.texture_dict.items()
                    if v.get("relative_path")
                },
                f,
            )

    def _track_prim_if_needed(
        self,
        prim: Usd.Prim,
        sim_name: str,
        prim_path: str,
        indent: int,
    ) -> None:
        rigid_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if rigid_attr and rigid_attr.Get():
            self.tracked_prims.append(
                {"name": sim_name, "prim": prim, "prim_path": prim_path}
            )

        deform_attr = prim.GetAttribute("physxDeformable:deformableEnabled")
        if deform_attr and deform_attr.Get():
            self.tracked_deform_prims.append(
                {"name": sim_name, "prim": prim, "prim_path": prim_path}
            )

    def _resolve_texture_path(self, mat_shader: UsdShade.Shader) -> Optional[str]:
        for input_name in ("diffuse_texture", "AlbedoTexture"):
            shader_input = mat_shader.GetInput(input_name)
            if not shader_input:
                continue
            value = shader_input.Get()
            if value is None:
                continue

            resolved_path = str(getattr(value, "resolvedPath", ""))
            if resolved_path:
                return resolved_path

            raw_path = str(getattr(value, "path", ""))
            if raw_path:
                return raw_path

        return None

    def _resolve_diffuse_color(self, mat_shader: UsdShade.Shader) -> List[float]:
        for input_name in ("diffuse_color_constant", "diffuseColor"):
            shader_input = mat_shader.GetInput(input_name)
            if not shader_input:
                continue
            value = shader_input.Get()
            if value is not None:
                return [float(value[0]), float(value[1]), float(value[2])]
        return [1.0, 1.0, 1.0]

    def _load_texture_image(self, texture_path: str) -> Optional[Image.Image]:
        if texture_path in self.texture_dict:
            tex_info = self.texture_dict[texture_path]
            tex_image = tex_info.get("image")
            tex_relative = tex_info.get("relative_path")
            if (
                tex_image is None
                and isinstance(tex_relative, str)
                and self.texture_cache_dir
            ):
                local_path = os.path.join(self.texture_cache_dir, tex_relative)
                if os.path.isfile(local_path):
                    tex_info["image"] = Image.open(local_path)
            cached_image = tex_info.get("image")
            return cached_image if isinstance(cached_image, Image.Image) else None

        image: Optional[Image.Image] = None
        if texture_path.startswith(("http://", "https://")):
            response = requests.get(texture_path, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        elif os.path.isfile(texture_path):
            image = Image.open(texture_path)

        if image is not None and self.texture_cache_dir is not None:
            ext = os.path.splitext(texture_path)[1] or ".png"
            texture_file_name = f"{uuid.uuid4()}{ext}"
            texture_file_path = os.path.join(self.texture_cache_dir, texture_file_name)
            image.save(texture_file_path)
            self.texture_dict[texture_path] = {
                "relative_path": texture_file_name,
                "image": image,
            }

        return image

    def _process_mesh_prim(
        self,
        prim: Usd.Prim,
        sim_obj: SimObject,
        indent: int,
        mat_info: Optional[MaterialContext],
    ) -> None:
        mesh_prim = UsdGeom.Mesh(prim)
        if not mesh_prim:
            return

        vertices = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices_orig = np.asarray(
            mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32
        )
        face_vertex_counts = np.asarray(
            mesh_prim.GetFaceVertexCountsAttr().Get(), dtype=np.int32
        )
        if face_vertex_counts.size == 0:
            return

        unique_counts = set(face_vertex_counts.tolist())
        if len(unique_counts) != 1:
            pyzlc.warning("Mixed face vertex counts are not supported; skipping mesh.")
            return

        num_vert_per_face = int(face_vertex_counts[0])
        if num_vert_per_face not in {3, 4}:
            pyzlc.warning(f"Unsupported face vertex count: {num_vert_per_face}")
            return

        indices = indices_orig.reshape(-1, num_vert_per_face)

        mesh_infos = self._collect_mesh_parts(
            prim, mesh_prim, vertices, indices, mat_info
        )
        for mesh_info in mesh_infos:
            texture_visual = None
            uv_buf = mesh_info.get("uv")
            if uv_buf is not None:
                texture_visual = trimesh.visual.TextureVisuals(uv=uv_buf)

            mesh_obj = trimesh.Trimesh(
                vertices=mesh_info["vertices"],
                faces=mesh_info["indices"],
                visual=texture_visual,
                process=False,
            )
            mesh_obj.fix_normals()
            trimesh.repair.fix_winding(mesh_obj)
            trimesh.repair.fix_inversion(mesh_obj, True)

            sim_visual = self.build_mesh_buffer(mesh_obj)
            if mesh_info["material"] is not None:
                sim_visual["material"] = mesh_info["material"]
            sim_obj["visuals"].append(sim_visual)

    def _collect_mesh_parts(
        self,
        prim: Usd.Prim,
        mesh_prim: UsdGeom.Mesh,
        vertices: np.ndarray,
        indices: np.ndarray,
        mat_info: Optional[MaterialContext],
    ) -> List[dict]:
        mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh_prim)
        if not mesh_subsets:
            uv_buf = self._resolve_mesh_uv(prim, indices, vertices, mat_info)
            return [
                {
                    "vertices": vertices,
                    "indices": indices,
                    "uv": uv_buf,
                    "material": mat_info[0] if mat_info is not None else None,
                }
            ]

        subset_uvs = self._collect_subset_uvs(prim)
        mesh_infos: List[dict] = []
        for subset in mesh_subsets:
            subset_mask = subset.GetIndicesAttr().Get()
            subset_indices = indices[subset_mask]
            uv_key = subset_indices.shape[0] * subset_indices.shape[1]

            subset_mat_info = self.process_prim_material(subset.GetPrim(), indent=0)
            subset_mat = (
                subset_mat_info
                if subset_mat_info is not None
                else (mat_info[0] if mat_info is not None else None)
            )

            mesh_infos.append(
                {
                    "vertices": vertices,
                    "indices": subset_indices,
                    "uv": subset_uvs.get(uv_key),
                    "material": subset_mat,
                }
            )

        return mesh_infos

    def _collect_subset_uvs(self, prim: Usd.Prim) -> Dict[int, np.ndarray]:
        subset_uvs: Dict[int, np.ndarray] = {}
        primvars = UsdGeom.PrimvarsAPI(prim)

        if primvars.HasPrimvar("st"):
            uvs = np.asarray(primvars.GetPrimvar("st").Get(), dtype=np.float32)
            subset_uvs[uvs.shape[0]] = uvs

        for i in range(1, 100):
            primvar_name = f"st_{i}"
            if not primvars.HasPrimvar(primvar_name):
                break
            uvs_more = np.asarray(
                primvars.GetPrimvar(primvar_name).Get(), dtype=np.float32
            )
            subset_uvs[uvs_more.shape[0]] = uvs_more

        return subset_uvs

    def _resolve_mesh_uv(
        self,
        prim: Usd.Prim,
        indices: np.ndarray,
        vertices: np.ndarray,
        mat_info: Optional[MaterialContext],
    ) -> Optional[np.ndarray]:
        primvars = UsdGeom.PrimvarsAPI(prim)
        if mat_info is not None and mat_info[1]:
            return self.compute_projected_uv(
                prim=prim,
                vertex_buf=vertices,
                index_buf=indices,
                use_world_coord=mat_info[2],
            )

        if primvars.HasPrimvar("st"):
            uvs = np.asarray(primvars.GetPrimvar("st").Get(), dtype=np.float32)
            if uvs.shape[0] == indices.shape[0] * indices.shape[1]:
                return uvs

        return None

    def _process_primitive_prim(
        self,
        prim_type: str,
        prim_path: str,
        mat_info: Optional[MaterialContext],
    ) -> Optional[SimVisual]:
        rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
        if not rt_prim or not rt_prim.IsValid():
            return None

        material = (
            mat_info[0] if mat_info is not None else create_material([1, 1, 1, 1])
        )
        identity = SimTransform(pos=[0, 0, 0], rot=[0, 0, 0, 1], scale=[1, 1, 1])

        if prim_type == "Cone":
            cone_prim = RtGeom.Cone(rt_prim)
            axis = cone_prim.GetAxisAttr().Get()
            height = float(cone_prim.GetHeightAttr().Get())
            radius = float(cone_prim.GetRadiusAttr().Get())

            cone_mesh = trimesh.creation.cone(height=height, radius=radius)
            cone_mesh.apply_transform(
                trimesh.transformations.translation_matrix([0, 0, -height * 0.5])
            )
            if axis == "Y":
                cone_mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
                )
            elif axis == "X":
                cone_mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
                )

            sim_visual = self.build_mesh_buffer(cone_mesh)
            sim_visual["material"] = material
            return sim_visual

        if prim_type == "Cube":
            cube_prim = RtGeom.Cube(rt_prim)
            cube_size = float(cube_prim.GetSizeAttr().Get())
            scale_attr = rt_prim.GetAttribute("xformOp:scale")
            cube_scale = [1.0, 1.0, 1.0]
            if scale_attr and scale_attr.IsValid():
                scale_val = scale_attr.Get()
                if scale_val is not None:
                    cube_scale = scale_val
            trans = SimTransform(
                pos=[0, 0, 0],
                rot=[0, 0, 0, 1],
                scale=[
                    cube_size * float(cube_scale[1]),
                    cube_size * float(cube_scale[2]),
                    cube_size * float(cube_scale[0]),
                ],
            )
            return SimVisual(
                name=f"{prim_path.replace('/', '_')}_visual_{prim_type}",
                type=VisualType.CUBE,
                mesh=None,
                trans=trans,
                material=material,
            )

        if prim_type == "Capsule":
            cap_prim = RtGeom.Capsule(rt_prim)
            axis = cap_prim.GetAxisAttr().Get()
            height = float(cap_prim.GetHeightAttr().Get())
            radius = float(cap_prim.GetRadiusAttr().Get())
            full_height = max(height + 2.0 * radius, 2.0 * radius)
            scale = {
                "X": [full_height, 2.0 * radius, 2.0 * radius],
                "Y": [2.0 * radius, full_height, 2.0 * radius],
                "Z": [2.0 * radius, 2.0 * radius, full_height],
            }.get(axis, [2.0 * radius, full_height, 2.0 * radius])
            return SimVisual(
                name=f"{prim_path.replace('/', '_')}_visual_{prim_type}",
                type=VisualType.CAPSULE,
                mesh=None,
                trans=SimTransform(
                    pos=identity["pos"], rot=identity["rot"], scale=scale
                ),
                material=material,
            )

        if prim_type == "Cylinder":
            cylinder_prim = RtGeom.Cylinder(rt_prim)
            axis = cylinder_prim.GetAxisAttr().Get()
            height = float(cylinder_prim.GetHeightAttr().Get())
            radius = float(cylinder_prim.GetRadiusAttr().Get())
            scale = {
                "X": [height, 2.0 * radius, 2.0 * radius],
                "Y": [2.0 * radius, height, 2.0 * radius],
                "Z": [2.0 * radius, 2.0 * radius, height],
            }.get(axis, [2.0 * radius, height, 2.0 * radius])
            return SimVisual(
                name=f"{prim_path.replace('/', '_')}_visual_{prim_type}",
                type=VisualType.CYLINDER,
                mesh=None,
                trans=SimTransform(
                    pos=identity["pos"], rot=identity["rot"], scale=scale
                ),
                material=material,
            )

        if prim_type == "Sphere":
            sphere_prim = RtGeom.Sphere(rt_prim)
            radius = float(sphere_prim.GetRadiusAttr().Get())
            return SimVisual(
                name=f"{prim_path.replace('/', '_')}_visual_{prim_type}",
                type=VisualType.SPHERE,
                mesh=None,
                trans=SimTransform(
                    pos=identity["pos"],
                    rot=identity["rot"],
                    scale=[radius * 2.0, radius * 2.0, radius * 2.0],
                ),
                material=material,
            )

        return None
