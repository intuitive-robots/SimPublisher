import io
import json
import math
import os
import timeit
import uuid
from collections import defaultdict
from dataclasses import dataclass
from hashlib import md5

import numpy as np
import numpy.typing as npt
import omni
import omni.usd
import requests
import trimesh
import trimesh.visual
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, UsdUtils
from tabulate import tabulate
from usdrt import Usd as RtUsd
from usdrt import UsdGeom as RtGeom

# support for IsaacSim versions < 4.5
from importlib.metadata import version
if version("isaacsim") < "4.5":
    from omni.isaac.core.prims import XFormPrim as SingleXFormPrim
    from omni.isaac.core.utils.rotations import quat_to_rot_matrix, euler_angles_to_quat
else:
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.rotations import quat_to_rot_matrix, euler_angles_to_quat

from ..parser.mesh_utils import Mesh as MeshData
from ..parser.mesh_utils import split_mesh_faces
from .simdata import (
    SimAsset,
    SimMaterial,
    SimMesh,
    SimObject,
    SimScene,
    SimTexture,
    SimTransform,
    SimVisual,
    VisualType,
)


class Timer:
    class TimerObject:
        def __init__(self, name, timer_obj):
            self.name = name
            self.accum_times = timer_obj.accum_times
            self.started_timers = timer_obj.started_timers
            self.start_time = None

        def __enter__(self):
            if self.name in self.started_timers:
                raise RuntimeError(f"repeatedly started timer: {self.name}")
            self.started_timers.add(self.name)
            self.start_time = timeit.default_timer()
            return self

        def __exit__(self, exception_type, exception_value, exception_traceback):
            self.accum_times[self.name] += timeit.default_timer() - self.start_time
            self.started_timers.remove(self.name)

    def __init__(self):
        self.accum_times = defaultdict(lambda: 0)
        self.started_timers = set()

    def start(self, name) -> TimerObject:
        return Timer.TimerObject(name, self)

    def print_timings(self):
        print("\n\n[*** timers (unit: seconds) ***]")
        print(tabulate(sorted(list(self.accum_times.items()), key=lambda x: x[0])))
        print()

    @staticmethod
    def time_function(name, timer_var="timer"):
        def decorator(func):
            def decorated_func(self, *args, **kwargs):
                with getattr(self, timer_var).start(name):
                    return func(self, *args, **kwargs)

            return decorated_func

        return decorator


@dataclass
class MaterialInfo:
    sim_mat: SimMaterial
    project_uvw: bool = False
    use_world_coord: bool = False

    def __post_init__(self):
        assert self.sim_mat is not None


@dataclass
class TextureInfo:
    relative_path: str = None
    image: Image = None


class IsaacSimStageParser:
    def __init__(
        self,
        stage: Usd.Stage,
        ignored_prim_paths: list[str] = [],
        texture_cache_dir: str = None,
    ) -> None:
        assert isinstance(stage, Usd.Stage)
        self.stage = stage
        self.ignored_prim_paths = set(ignored_prim_paths)

        # create a usdrt stage from a plain usd stage.
        # only usdrt api can access updated object states from fabric.
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        stage_id = stage_id.ToLongInt()
        print("usd stage id:", stage_id)

        self.rt_stage = RtUsd.Stage.Attach(stage_id)
        print("usdrt stage:", self.rt_stage)

        # usd prims whose poses are tracked
        self.tracked_prims: list[dict] = []
        self.tracked_deform_prims: list[dict] = []

        # timer logs
        self.timer = Timer()

        # cache for downloaded textures
        self.texture_cache_dir = texture_cache_dir
        # full texture path --> texture info
        self.texture_dict: dict[str, TextureInfo] = {}
        # path to the json file storing texture_dict
        self.texture_dict_path = None

        # load texture cache...
        if texture_cache_dir is not None:
            if not os.path.isdir(texture_cache_dir):
                os.makedirs(texture_cache_dir, exist_ok=True)
                print(f"texture cache dir [{texture_cache_dir}] does not exist. it is created.")
            print(f"using texture cache dir: {texture_cache_dir}")
            # load texture dict
            texture_dict_path = os.path.join(texture_cache_dir, "texture_dict.json")
            if os.path.isfile(texture_dict_path):
                print(f"loading texture dict from: {texture_dict_path}")
                with open(texture_dict_path) as f:
                    self.texture_dict = json.load(f)
                    # stores path relative to the texture_cache_dir
                    for k, v in self.texture_dict.items():
                        self.texture_dict[k] = TextureInfo(relative_path=v)
            self.texture_dict_path = texture_dict_path
        else:
            print("no texture cache dir is specified; performance will degrade.")

    def get_usdrt_stage(self) -> RtUsd.Stage:
        return self.rt_stage

    def get_tracked_prims(self) -> tuple[list[dict], list[dict]]:
        return self.tracked_prims, self.tracked_deform_prims

    def print_timers(self):
        self.timer.print_timings()

    def parse_scene(self) -> SimScene:
        print("parsing stage:", self.stage)

        scene = SimScene()
        scene.root = SimObject(name="root", trans=SimTransform())
        self.sim_scene = scene

        # parse the usd stage
        root_path = "/World"
        with self.timer.start("parse_prim_tree"):
            sim_obj = self.parse_prim_tree(root=self.stage.GetPrimAtPath(root_path))

        assert sim_obj is not None
        scene.root.children.append(sim_obj)

        # show timing information
        self.print_timers()

        # store texture dict
        if self.texture_dict_path is not None:
            with open(self.texture_dict_path, "w") as f:
                json.dump({k: v.relative_path for k, v in self.texture_dict.items()}, f)
            print(f"texture dict stored to: {self.texture_dict_path}")

        return scene

    def parse_prim_tree(
        self,
        root: Usd.Prim,
        indent=0,
        parent_path=None,
        inherited_material: MaterialInfo | None = None,
    ) -> SimObject | None:
        """parse the tree starting from a prim"""

        if str(root.GetPath()) in self.ignored_prim_paths:
            return

        # define prim types to be handled
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
            # TODO: traverse twice and preserve only prims with meshes as children
            return

        # filter out colliders prims
        purpose_attr = root.GetAttribute("purpose")
        if purpose_attr and purpose_attr.Get() in {"proxy", "guide"}:
            return

        # compute usd path of current prim
        if parent_path is None:
            prim_path = str(root.GetPrimPath())
        else:
            prim_path = f"{parent_path}/{root.GetName()}"

        # compute local transforms
        translate, rot, scale = self.compute_local_trans(root)

        # create a node for current prim
        sim_object = SimObject(
            name=prim_path.replace("/", "_"),
            trans=SimTransform(pos=translate, rot=rot, scale=scale),
        )

        print(
            "\t" * indent + f"{prim_path}: {root.GetTypeName()} "
            f"{root.GetAttribute('purpose').Get()} "
            f"{sim_object.trans.scale}"
        )

        # parse material
        mat_info = self.parse_prim_material(prim=root, indent=indent)
        # print("\t" * indent + f"material parsed: {mat_info}")

        # parse meshes and other primitive shapes
        self.parse_prim_geometries(
            prim=root,
            prim_path=prim_path,
            sim_obj=sim_object,
            indent=indent,
            mat_info=mat_info or inherited_material,
        )

        # track prims with rigid objects attached
        if (attr := root.GetAttribute("physics:rigidBodyEnabled")) and attr.Get():
            print("\t" * indent + f"tracking {prim_path}")
            self.tracked_prims.append({"name": sim_object.name, "prim": root, "prim_path": prim_path})

        # track prims with deformable enabled
        if (attr := root.GetAttribute("physxDeformable:deformableEnabled")) and attr.Get():
            print("\t" * indent + f"tracking deform {prim_path}")
            self.tracked_deform_prims.append({"name": sim_object.name, "prim": root, "prim_path": prim_path})

        child: Usd.Prim
        if root.IsInstance():
            # handle the case where root is an instance of a prototype
            proto = root.GetPrototype()
            print("\t" * indent + f"@prototype: {proto.GetPrimPath()}")

            # parse child prims of the prototype
            for child in proto.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child,
                    indent=indent + 1,
                    parent_path=prim_path,
                    inherited_material=mat_info or inherited_material,
                ):
                    sim_object.children.append(obj)

        else:
            # parse child prims of the current prim (root)
            for child in root.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child,
                    indent=indent + 1,
                    parent_path=prim_path,
                    inherited_material=mat_info or inherited_material,
                ):
                    sim_object.children.append(obj)

        return sim_object

    def compute_local_trans(self, prim: Usd.Prim):
        # not really necessary...
        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

        # extract local transformation
        sc, rt, rto, tr = omni.usd.get_local_transform_SRT(prim, timecode)

        # reorder scale for unity coord system
        scale = [sc[1], sc[2], sc[0]]

        # reorder translate for unity coord system
        translate = [tr[1], tr[2], -tr[0]]

        # convert rot to quad
        # result order: (w, x, y, z)
        rtq = euler_angles_to_quat([rt[rto[0]], rt[rto[1]], rt[rto[2]]], True)
        # reorder rot for unity coord system
        rot = [-rtq[2], -rtq[3], rtq[1], rtq[0]]

        return translate, rot, scale

    def compute_world_trans(self, prim: Usd.Prim) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # TODO: use isaacsim api instead of usd api for getting transformations

        prim = SingleXFormPrim(str(prim.GetPath()))
        assert prim.is_valid()

        pos, quat = prim.get_world_pose()
        scale = prim.get_world_scale()

        return (
            pos.cpu().numpy(),
            quat_to_rot_matrix(quat.cpu().numpy()),
            scale.cpu().numpy(),
        )

    @Timer.time_function("parse_prim_material")
    def parse_prim_material(
        self,
        prim: Usd.Prim,
        indent: int,
    ) -> MaterialInfo | None:
        with self.timer.start("parse_prim_material_1"):
            matapi = UsdShade.MaterialBindingAPI(prim)
            if matapi is None:
                # print("\t" * indent + "material binding api not found")
                return

            mat = matapi.GetDirectBinding().GetMaterial()
            if not mat:
                # print("\t" * indent + "material not found")
                return

            mat_prim = self.stage.GetPrimAtPath(mat.GetPath())
            if not mat_prim:
                # print("\t" * indent + "material prim not found")
                return

            if not mat_prim.GetAllChildren():
                # print("\t" * indent + "material has no shaders")
                return

            # we only care about the first shader
            mat_shader_prim = mat_prim.GetAllChildren()[0]
            mat_shader = UsdShade.Shader(mat_shader_prim)

        with self.timer.start("parse_prim_material_2"):
            diffuse_texture = mat_shader.GetInput("diffuse_texture")
            texture_path = None
            if diffuse_texture.Get() is not None:
                texture_path = str(diffuse_texture.Get().resolvedPath)
                if not texture_path:
                    texture_path = str(diffuse_texture.Get().path)
            else:
                diffuse_texture = mat_shader.GetInput("AlbedoTexture")
                if diffuse_texture.Get() is not None:
                    texture_path = str(diffuse_texture.Get().resolvedPath)
                    if not texture_path:
                        texture_path = str(diffuse_texture.Get().path)

        with self.timer.start("parse_prim_material_3"):
            diffuse_color = [1.0, 1.0, 1.0]
            if (c := mat_shader.GetInput("diffuse_color_constant").Get()) is not None:
                diffuse_color = [c[0], c[1], c[2]]
            elif (c := mat_shader.GetInput("diffuseColor").Get()) is not None:
                diffuse_color = [c[0], c[1], c[2]]

            sim_mat = SimMaterial(color=diffuse_color + [1])

        if texture_path is not None:
            with self.timer.start("parse_prim_material_4.1"):
                image = None
                # first try to find the texture in cache
                if texture_path in self.texture_dict:
                    tex_info = self.texture_dict[texture_path]
                    if tex_info.image is None:
                        tex_info.image = Image.open(os.path.join(self.texture_cache_dir, tex_info.relative_path))
                    image = tex_info.image

                # if not found, download the texture and add it to cache
                else:
                    if texture_path.startswith(("http://", "https://")):
                        response = requests.get(texture_path)
                        image = Image.open(io.BytesIO(response.content))
                    elif os.path.isfile(texture_path):
                        image = Image.open(texture_path)

                    # this is not needed for local textures. but anyway...
                    if image is not None and self.texture_cache_dir is not None:
                        ext = os.path.splitext(texture_path)[1]
                        texture_file_name = f"{str(uuid.uuid4())}{ext}"
                        texture_file_path = os.path.join(self.texture_cache_dir, texture_file_name)
                        image.save(texture_file_path)
                        self.texture_dict[texture_path] = TextureInfo(relative_path=texture_file_name, image=image)

            with self.timer.start("parse_prim_material_4.2"):
                if image is not None:
                    image = image.convert("RGB")
                    image = np.array(image).astype(np.uint8)
                    image, height, width = SimTexture.compress_image(image, height=image.shape[0], width=image.shape[1])
                    # sim_mat.texture = SimTexture.create_texture(
                    #     np.array(image),
                    #     height=image.height,
                    #     width=image.width,
                    #     scene=self.sim_scene,
                    # )
                    bin_data = image.tobytes()
                    # assert len(bin_data) == image.width * image.height * 3
                    tex_hash = SimTexture.generate_hash(bin_data)
                    sim_mat.texture = SimTexture(
                        hash=tex_hash,
                        width=image.shape[1],
                        height=image.shape[0],
                        textureType="2D",
                        textureScale=(1, 1),
                    )
                    self.sim_scene.raw_data[tex_hash] = bin_data
                    # sim_mat.texture.compress(self.sim_scene.raw_data)

        with self.timer.start("parse_prim_material_5"):
            mi = MaterialInfo(sim_mat=sim_mat)

            if (use_uvw := mat_shader.GetInput("project_uvw").Get()) is not None and use_uvw is True:
                mi.project_uvw = True

                if (world_coord := mat_shader.GetInput("world_or_object").Get()) is not None and world_coord is True:
                    mi.use_world_coord = True

        return mi

    def compute_projected_uv(self, prim: Usd.Prim, vertex_buf, index_buf, use_world_coord: bool = False):
        """project_uvw: cube map for uv"""

        assert type(vertex_buf) is np.ndarray
        assert len(vertex_buf.shape) == 2 and vertex_buf.shape[1] in {3, 4}
        assert type(index_buf) is np.ndarray
        assert len(index_buf.shape) == 2 and index_buf.shape[1] in {3, 4}

        uvs = []
        axes = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        axis_projectors = [
            lambda v: [v[0], v[1]],
            lambda v: [v[0], -v[1]],
            lambda v: [v[0], v[2]],
            lambda v: [-v[0], v[2]],
            lambda v: [v[1], v[2]],
            lambda v: [-v[1], v[2]],
        ]

        for tri in index_buf:
            p0 = vertex_buf[tri[0]]
            p1 = vertex_buf[tri[1]]
            p2 = vertex_buf[tri[2]]
            if len(tri) == 4:
                p3 = vertex_buf[tri[3]]

            if use_world_coord:
                pos, quat, scale = self.compute_world_trans(prim)
                p0 = quat @ (p0 * scale) + pos
                p1 = quat @ (p1 * scale) + pos
                p2 = quat @ (p2 * scale) + pos
                if len(tri) == 4:
                    p3 = quat @ (p3 * scale) + pos

            normal = np.cross(p1 - p0, p2 - p0)
            normal /= np.linalg.norm(normal)

            axis_id = np.argmax(axes @ normal)
            projector = axis_projectors[axis_id]
            uvs.append(projector(p0))
            uvs.append(projector(p1))
            uvs.append(projector(p2))
            if len(tri) == 4:
                uvs.append(projector(p3))

        assert len(uvs) == index_buf.shape[0] * index_buf.shape[1]
        return np.array(uvs)

    @Timer.time_function("parse_prim_geometries")
    def parse_prim_geometries(
        self,
        prim: Usd.Prim,
        prim_path: str,
        sim_obj: SimObject,
        indent: int,
        mat_info: MaterialInfo | None = None,
    ):
        prim_type = prim.GetTypeName()

        # check visibility first
        if str(prim.GetAttribute("visibility").Get()) == "invisible":
            return

        if prim_type == "Mesh":
            with self.timer.start("parse_prim_geometries_mesh_1"):
                # currently each instance of a prototype will create a different mesh object
                # detecting this and use the same mesh object would reduce memory usage

                # for soft body, maybe use usdrt.UsdGeom.xxx (in get_update() function, not here)
                mesh_prim = UsdGeom.Mesh(prim)
                assert mesh_prim is not None

                # read vertices, normals and indices

                vertices = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
                # normals = np.asarray(mesh_prim.GetNormalsAttr().Get(), dtype=np.float32)
                indices_orig = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                face_vertex_counts = np.asarray(mesh_prim.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

                # assuming there are either only triangular faces or only quad faces...
                assert len(set(face_vertex_counts)) == 1
                num_vert_per_face = face_vertex_counts[0]
                assert num_vert_per_face in {3, 4}
                indices = indices_orig.reshape(-1, num_vert_per_face)
                # if indices.shape[0] * indices.shape[1] != normals.shape[0]:
                #     raise RuntimeError(
                #         f"indices shape {indices.shape} and normals shape {normals.shape} mismatch. "
                #         f"vertices shape: {vertices.shape}"
                #     )

                # get uv coordinates and store mesh data

                mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh_prim)
                mesh_info_list = []

            # if the mesh has multiple GeomSubsets
            if mesh_subsets:
                with self.timer.start("parse_prim_geometries_mesh_2.1"):
                    # retrieve uv sets for GeomSubsets
                    subset_uvs = {}
                    if UsdGeom.PrimvarsAPI(prim).HasPrimvar("st"):
                        uvs = np.asarray(
                            UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(),
                            dtype=np.float32,
                        )
                        subset_uvs[uvs.shape[0]] = uvs

                        for i in range(1, 100):
                            if UsdGeom.PrimvarsAPI(prim).HasPrimvar(f"st_{i}"):
                                uvs_more = np.asarray(
                                    UsdGeom.PrimvarsAPI(prim).GetPrimvar(f"st_{i}").Get(),
                                    dtype=np.float32,
                                )
                                subset_uvs[uvs_more.shape[0]] = uvs_more
                            else:
                                break

                # process GeomSubsets
                with self.timer.start("parse_prim_geometries_mesh_2.2"):
                    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh_prim):
                        # get subset indices
                        subset_mask = subset.GetIndicesAttr().Get()
                        subset_indices = indices[subset_mask]
                        # subset_normals = np.array(
                        #     [normals[j * num_vert_per_face + i] for i in range(num_vert_per_face) for j in subset_mask]
                        # )

                        # get subset material
                        # TODO: handle project_uvw?
                        subset_mat_info = self.parse_prim_material((subset), indent + 1)
                        subset_mat = None
                        if subset_mat_info is not None:
                            subset_mat = subset_mat_info.sim_mat
                        elif mat_info is not None:
                            subset_mat = mat_info.sim_mat

                        mesh_info_list.append(
                            {
                                "mesh": MeshData(
                                    vertex_buf=vertices,
                                    # normal_buf=subset_normals,
                                    index_buf=subset_indices,
                                    uv_buf=subset_uvs.get(
                                        subset_indices.shape[0] * subset_indices.shape[1],
                                        None,
                                    ),
                                ),
                                "material": subset_mat,
                            }
                        )

            # if the mesh has no GeomSubsets
            else:
                with self.timer.start("parse_prim_geometries_mesh_3"):
                    uvs = None
                    # get uv: compute projected uv with cube mapping
                    if mat_info is not None and mat_info.project_uvw:
                        # [!] this will compute uv per-index, NOT per-vertex
                        uvs = self.compute_projected_uv(
                            prim=prim,
                            vertex_buf=vertices,
                            index_buf=indices,
                            use_world_coord=mat_info.use_world_coord,
                        )
                    elif UsdGeom.PrimvarsAPI(prim).HasPrimvar("st"):
                        uvs = np.asarray(
                            UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(),
                            dtype=np.float32,
                        )
                        # discard invalid uv buffer
                        if uvs.shape[0] != indices.shape[0] * indices.shape[1]:
                            uvs = None

                    # if the mesh
                    mesh_info_list.append(
                        {
                            "mesh": MeshData(
                                vertex_buf=vertices,
                                # normal_buf=normals,
                                index_buf=indices,
                                uv_buf=uvs,
                            ),
                            "material": mat_info.sim_mat if mat_info is not None else None,
                        }
                    )

            # create SimMesh objects
            for mesh_info in mesh_info_list:
                # [!] here we expect the uvs to be already per-index

                # TODO: Fix whatever this is supposed to do so it doenst create ghost vertices
                # with self.timer.start("parse_prim_geometries_mesh_4"):
                #     mesh_data = split_mesh_faces(mesh_info["mesh"])
                
                # after split_mesh_faces is fixed this can be removed
                mesh_data = mesh_info["mesh"]

                with self.timer.start("parse_prim_geometries_mesh_5"):
                    texture_visual = None
                    if mesh_data.uv_buf is not None:
                        texture_visual = trimesh.visual.TextureVisuals(uv=mesh_data.uv_buf)

                    mesh_obj = trimesh.Trimesh(
                        vertices=mesh_data.vertex_buf,
                        # vertex_normals=mesh_data.normal_buf,
                        faces=mesh_data.index_buf,
                        visual=texture_visual,
                        # can' process, otherwise deformable object meshes have to be processed every time they
                        # are transmitted. 
                        process=False,
                    )
                    mesh_obj.fix_normals()
                    trimesh.repair.fix_winding(mesh_obj)
                    trimesh.repair.fix_inversion(mesh_obj, True)

                    print("\t" * (indent + 1) + "[mesh geometry]")
                    print("\t" * (indent + 1) + f"vertex:   {mesh_obj.vertices.shape}")
                    print("\t" * (indent + 1) + f"normal:   {mesh_obj.vertex_normals.shape}")
                    print("\t" * (indent + 1) + f"index:    {mesh_obj.faces.shape}")
                    if mesh_data.uv_buf is not None:
                        print("\t" * (indent + 1) + f"uv:       {mesh_obj.visual.uv.shape}")

                # # #######################################################################################
                # # (for debug) export extracted mesh and check
                # if mesh_data.index_buf.shape[1] == 3:
                #     # import open3d as o3d

                #     # o3dverts = o3d.utility.Vector3dVector(mesh_obj.vertices)
                #     # o3dtris = o3d.utility.Vector3iVector(mesh_obj.faces.astype(np.int64))

                #     # mesh_np = o3d.geometry.TriangleMesh(o3dverts, o3dtris)
                #     # mesh_np.compute_vertex_normals()

                #     # # mesh_np.vertex_colors = o3d.utility.Vector3dVector(
                #     # #     np.random.uniform(0, 1, size=(5, 3))
                #     # # )
                #     # # mesh_np.compute_vertex_normals()
                #     # o3d.visualization.draw_geometries([mesh_np])
                #     # # o3d.io.write_triangle_mesh("./a.obj", mesh_np)
                #     # # raise SystemError()

                #     # # mesh_obj.vertices = np.asarray(mesh_np.vertices)
                #     # # mesh_obj.faces = np.asarray(mesh_np.triangles)
                #     # # mesh_obj.vertex_normals = np.asarray(mesh_np.vertex_normals)

                #     with open(f"./{prim_path.replace('/','-')}-{uuid.uuid4().hex}.obj", "w") as f:
                #         for v in mesh_obj.vertices:
                #             f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                #         for vn in mesh_obj.vertex_normals:
                #             f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
                #         for i in mesh_obj.faces:
                #             f.write(f"f {i[0]+1} {i[1]+1} {i[2]+1}\n")
                #     # raise SystemError

                # # #######################################################################################

                with self.timer.start("parse_prim_geometries_mesh_6"):
                    sim_mesh = self.build_mesh_buffer(mesh_obj)

                    if mesh_info["material"] is not None:
                        sim_mesh.material = mesh_info["material"]
                        print("\t" * (indent + 1) + f"material: {sim_mesh.material}")

                    sim_obj.visuals.append(sim_mesh)

        elif prim_type == "Cube":
            with self.timer.start("parse_prim_geometries_cube"):
                rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
                # only handle scale...
                # translation: xformOp:translate
                # rotation: xformOp:orient
                cube_scale = rt_prim.GetAttribute("xformOp:scale").Get()

                cube_prim = RtGeom.Cube(rt_prim)
                cube_size = cube_prim.GetSizeAttr().Get()

                sim_cube = SimVisual(
                    name="Visual_Cube",
                    type=VisualType.CUBE,
                    trans=SimTransform(
                        scale=[
                            cube_size * cube_scale[1],
                            cube_size * cube_scale[2],
                            cube_size * cube_scale[0],
                        ]
                    ),
                    material=SimMaterial(color=[1.0, 1.0, 1.0, 1.0]),
                )

                if mat_info is not None:
                    print("\t" * indent + f"material: {mat_info}")
                    sim_cube.material = mat_info.sim_mat
                sim_obj.visuals.append(sim_cube)
                sim_obj.trans.scale = [1.0] * 3

        elif prim_type == "Capsule":
            with self.timer.start("parse_prim_geometries_capsule"):
                rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
                cap_prim = RtGeom.Capsule(rt_prim)

                axis = cap_prim.GetAxisAttr().Get()
                height = cap_prim.GetHeightAttr().Get()
                radius = cap_prim.GetRadiusAttr().Get()

                capsule_mesh = trimesh.creation.capsule(height=height, radius=radius)
                if axis == "Y":
                    capsule_mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
                elif axis == "X":
                    capsule_mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))

                # scale/translation/rotation not handled,
                # since it seems that isaac lab won't modify them...

                sim_mesh = self.build_mesh_buffer(capsule_mesh)
                if mat_info is not None:
                    print("\t" * indent + f"material: {mat_info}")
                    sim_mesh.material = mat_info.sim_mat
                sim_obj.visuals.append(sim_mesh)

        elif prim_type == "Cone":
            with self.timer.start("parse_prim_geometries_cone"):
                rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
                cap_prim = RtGeom.Cone(rt_prim)

                axis = cap_prim.GetAxisAttr().Get()
                height = cap_prim.GetHeightAttr().Get()
                radius = cap_prim.GetRadiusAttr().Get()

                cone_mesh = trimesh.creation.cone(height=height, radius=radius)
                cone_mesh.apply_transform(trimesh.transformations.translation_matrix([0, 0, -height * 0.5]))
                if axis == "Y":
                    cone_mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
                elif axis == "X":
                    cone_mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))

                # scale/translation/rotation not handled,
                # since it seems that isaac lab won't modify them...

                sim_mesh = self.build_mesh_buffer(cone_mesh)
                if mat_info is not None:
                    print("\t" * indent + f"material: {mat_info}")
                    sim_mesh.material = mat_info.sim_mat
                sim_obj.visuals.append(sim_mesh)

        elif prim_type == "Cylinder":
            with self.timer.start("parse_prim_geometries_cylinder"):
                rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
                cap_prim = RtGeom.Cylinder(rt_prim)

                axis = cap_prim.GetAxisAttr().Get()
                height = cap_prim.GetHeightAttr().Get()
                radius = cap_prim.GetRadiusAttr().Get()

                cylinder_mesh = trimesh.creation.cylinder(height=height, radius=radius)
                if axis == "Y":
                    cylinder_mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
                elif axis == "X":
                    cylinder_mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))

                # scale/translation/rotation not handled,
                # since it seems that isaac lab won't modify them...

                sim_mesh = self.build_mesh_buffer(cylinder_mesh)
                if mat_info is not None:
                    print("\t" * indent + f"material: {mat_info}")
                    sim_mesh.material = mat_info.sim_mat
                sim_obj.visuals.append(sim_mesh)

        elif prim_type == "Sphere":
            with self.timer.start("parse_prim_geometries_sphere"):
                rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
                cap_prim = RtGeom.Sphere(rt_prim)

                radius = cap_prim.GetRadiusAttr().Get()

                sphere_mesh = trimesh.creation.uv_sphere(radius=radius)

                # scale/translation/rotation not handled,
                # since it seems that isaac lab won't modify them...

                sim_mesh = self.build_mesh_buffer(sphere_mesh)
                if mat_info is not None:
                    print("\t" * indent + f"material: {mat_info}")
                    sim_mesh.material = mat_info.sim_mat
                sim_obj.visuals.append(sim_mesh)

    @Timer.time_function("build_mesh_buffer")
    def build_mesh_buffer(self, mesh_obj: trimesh.Trimesh):
        # rotate mesh to match unity coord system
        rot_mat = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        mesh_obj.apply_transform(rot_mat)

        # this will create smooth vertex normals.
        # in isaac sim the same vertex on different faces can have different normals,
        # but this is not supported by simpub, so here per-vertex normals are calculated.
        mesh_obj.fix_normals()

        # fill some buffers
        bin_buffer = io.BytesIO()

        # Vertices
        verts = mesh_obj.vertices.astype(np.float32)
        verts = verts.flatten()
        vertices_layout = SimAsset.write_to_buffer(bin_buffer, verts)

        # Indices
        indices = mesh_obj.faces.astype(np.int32)
        indices = indices.flatten()
        indices_layout = SimAsset.write_to_buffer(bin_buffer, indices)

        # Normals
        normals = mesh_obj.vertex_normals.astype(np.float32)
        normals = normals.flatten()
        normal_layout = SimAsset.write_to_buffer(bin_buffer, normals)

        # Texture coords
        uv_layout = (0, 0)
        if hasattr(mesh_obj.visual, "uv") and mesh_obj.visual.uv is not None:
            uvs = mesh_obj.visual.uv.astype(np.float32)
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs = uvs.flatten()
            uv_layout = SimAsset.write_to_buffer(bin_buffer, uvs)

        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()

        #! todo: do not create new mesh when multiple primitives point to the same prototype
        mesh = SimMesh(
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            normalsLayout=normal_layout,
            uvLayout=uv_layout,
            hash=hash,
        )

        assert self.sim_scene is not None
        # self.sim_scene.meshes.append(mesh)
        self.sim_scene.raw_data[mesh.hash] = bin_data

        sim_mesh = SimVisual(
            name=mesh.hash,
            type=VisualType.MESH,
            mesh=mesh,
            material=SimMaterial(color=[1.0, 1.0, 1.0, 1.0]),
            trans=SimTransform(),
        )
        return sim_mesh
