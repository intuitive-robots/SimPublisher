import io
from hashlib import md5
import random
import math

import numpy as np
import trimesh

from simpub.core.simpub_server import SimPublisher
from simpub.simdata import (
    SimScene,
    SimObject,
    SimVisual,
    VisualType,
    SimTransform,
    SimMesh,
)

import omni
import omni.usd
from pxr import Usd, UsdGeom, Gf


class IsaacSimPublisher(SimPublisher):
    def __init__(self, host: str, stage: Usd.Stage) -> None:
        self.tracked_prims: list[dict] = []
        sim_scene = self.parse_scene(stage)
        super().__init__(sim_scene, host)

    def parse_scene(self, stage: Usd.Stage) -> SimScene:
        print("=" * 50)
        print("parsing stage:", stage)

        self.use_usdrt_stage(stage)

        scene = SimScene()
        self.sim_scene = scene

        scene.root = SimObject(name="root", trans=SimTransform())

        obj1 = SimObject(name="object_1", trans=SimTransform(pos=[10, 0, 0]))
        obj1.visuals.append(
            SimVisual(
                type=VisualType.CUBE,
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(),
            )
        )
        scene.root.children.append(obj1)

        bin_buffer = io.BytesIO()

        mesh = trimesh.creation.box(extents=[1, 2, 3])

        indices = mesh.faces.astype(np.int32)
        bin_buffer = io.BytesIO()

        # Vertices
        verts = mesh.vertices.astype(np.float32)
        verts[:, 2] = -verts[:, 2]
        verts = verts.flatten()
        vertices_layout = bin_buffer.tell(), verts.shape[0]
        bin_buffer.write(verts)

        # Normals
        norms = mesh.vertex_normals.astype(np.float32)
        norms[:, 2] = -norms[:, 2]
        norms = norms.flatten()
        normal_layout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms)

        # Indices
        indices = mesh.faces.astype(np.int32)
        indices = indices[:, [2, 1, 0]]
        indices = indices.flatten()
        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)

        # # Texture coords
        # uv_layout = (0, 0)
        # if hasattr(mesh.visual, "uv"):
        #     uvs = mesh.visual.uv.astype(np.float32)
        #     uvs[:, 1] = 1 - uvs[:, 1]
        #     uvs = uvs.flatten()
        #     uv_layout = bin_buffer.tell(), uvs.shape[0]

        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()

        mesh = SimMesh(
            id="mesh_1",
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            dataHash=hash,
            normalsLayout=normal_layout,
            uvLayout=(0, 0),
        )

        self.sim_scene.meshes.append(mesh)
        self.sim_scene.raw_data[mesh.dataHash] = bin_data

        obj2 = SimObject(name="object_2")
        obj2.visuals.append(
            SimVisual(
                type=VisualType.MESH,
                mesh="mesh_1",
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(pos=[-10, 0, 0]),
            )
        )
        scene.root.children.append(obj2)

        # root_path = "/World/Origin1"
        root_path = "/World"
        obj2 = self.parse_prim_tree(root=stage.GetPrimAtPath(root_path))
        assert obj2 is not None
        scene.root.children.append(obj2)

        return scene

    def use_usdrt_stage(self, stage: Usd.Stage):
        import omni
        import omni.usd

        from pxr import Usd, UsdUtils
        from usdrt import Usd as RtUsd
        from usdrt import UsdGeom as RtGeom
        from usdrt import Rt

        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        stage_id = stage_id.ToLongInt()
        print("usdrt stage id:", stage_id)

        rtstage = RtUsd.Stage.Attach(stage_id)
        print("usdrt stage:", rtstage)

        self.rt_stage = rtstage

    def parse_prim_tree(
        self,
        root: Usd.Prim,
        indent=0,
        parent_path=None,
    ) -> SimObject | None:
        # # handle cube and return
        # # if the cube has children, they will be ignored...
        # if root.GetTypeName() == "Cube":
        #     prim_mesh_path = "/World/Origin1/Cube2/geometry/mesh"
        #     prim = stage.GetPrimAtPath(prim_mesh_path)
        #     # prim = usdrt.UsdGeom.Cube(prim)
        #     print(prim)
        #     # print(prim.GetSizeAttr().Get())

        #     # only handle scale...
        #     # translation: xformOp:translate
        #     # rotation: xformOp:orient
        #     print(prim.GetAttribute("xformOp:scale").Get())
        #     return

        if root.GetTypeName() not in {"Xform", "Mesh", "Scope", ""}:  # Cube
            # not good...
            # perhaps traverse twice and preserve only prims with meshes as children
            return

        purpose_attr = root.GetAttribute("purpose")
        if purpose_attr and purpose_attr.Get() in {"proxy", "guide"}:
            return

        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

        trans_mat = omni.usd.get_local_transform_matrix(root, timecode)

        # row_y, row_z = trans_mat.GetRow(1), trans_mat.GetRow(2)
        # trans_mat.SetRow(1, row_z)
        # trans_mat.SetRow(2, row_y)
        # col_y, col_z = trans_mat.GetColumn(1), trans_mat.GetColumn(2)
        # trans_mat.SetColumn(1, col_z)
        # trans_mat.SetColumn(2, col_y)

        # print(trans_mat.IsLeftHanded(), trans_mat.IsRightHanded())
        # print(trans_mat)

        x_scale = Gf.Vec3d(
            trans_mat[0][0], trans_mat[0][1], trans_mat[0][2]
        ).GetLength()
        y_scale = Gf.Vec3d(
            trans_mat[1][0], trans_mat[1][1], trans_mat[1][2]
        ).GetLength()
        z_scale = Gf.Vec3d(
            trans_mat[2][0], trans_mat[2][1], trans_mat[2][2]
        ).GetLength()
        scale = [x_scale, z_scale, y_scale]
        # print("\t" * indent, x_scale, y_scale, z_scale)

        translate = trans_mat.ExtractTranslation()
        translate = [-translate[1], translate[2], translate[0]]

        rot = trans_mat.ExtractRotationQuat()
        imag = rot.GetImaginary()
        rot = [imag[1], -imag[2], -imag[0], rot.GetReal()]

        if parent_path is None:
            prim_path = str(root.GetPrimPath())
        else:
            prim_path = f"{parent_path}/{root.GetName()}"

        sim_object = SimObject(
            name=prim_path.replace("/", "_"),
            trans=SimTransform(pos=translate, rot=rot, scale=scale),
        )

        print(
            "\t" * indent
            + f"{prim_path}: {root.GetTypeName()} {root.GetAttribute('purpose').Get()}"
        )

        # maybe time_code is necessary
        # trans_mat = omni.usd.get_local_transform_matrix(root)
        # print("\t" * indent + f"{trans_mat}")

        # attr: Usd.Property
        # for attr in root.GetProperties():
        #     print("\t" * indent + f"{attr.GetName()}")

        if root.GetTypeName() == "Mesh":
            mesh_prim = UsdGeom.Mesh(root)
            assert mesh_prim is not None

            points = np.asarray(mesh_prim.GetPointsAttr().Get()).astype(np.float32)
            # points[:, [1, 2]] = points[:, [2, 1]]

            normals = np.asarray(mesh_prim.GetNormalsAttr().Get()).astype(np.float32)
            # normals[:, [1, 2]] = normals[:, [2, 1]]

            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).astype(
                np.int32
            )

            face_vertex_counts = np.asarray(
                mesh_prim.GetFaceVertexCountsAttr().Get()
            ).astype(np.int32)

            # either only triangular faces or only quad faces
            assert len(set(face_vertex_counts)) == 1
            num_vert_per_face = face_vertex_counts[0]

            mesh_obj = trimesh.Trimesh(
                vertices=points, faces=indices.reshape(-1, num_vert_per_face)
            )
            mesh_obj = mesh_obj.apply_transform(
                trimesh.transformations.euler_matrix(-math.pi / 2.0, math.pi / 2.0, 0)
            )
            mesh_obj.fix_normals()

            points = points.flatten()
            normals = normals.flatten()
            indices = indices.flatten()

            print(
                "\t" * indent
                + f"vertices size: {points.shape[0] // 3} {points.shape[0] % 3}"
            )
            print(
                "\t" * indent
                + f"normals size: {normals.shape[0] // 3} {normals.shape[0] % 3}"
            )
            print(
                "\t" * indent
                + f"triangles: {indices.shape[0] // 3} {indices.shape[0] % 3}"
            )

            assert normals.shape[0] // 3 == indices.shape[0]
            print(
                "\t" * indent
                + f"normal per index: {normals.shape[0] // 3} {indices.shape[0]}"
            )
            # print(
            #     "\t" * indent + f"@vert {points} {points.dtype} {points.shape}"
            # )
            # print(
            #     "\t" * indent + f"@indi {indices} {indices.dtype} {indices.shape}"
            # )

            # bin_buffer = io.BytesIO()

            # # Vertices
            # vertices_layout = bin_buffer.tell(), points.shape[0]
            # bin_buffer.write(points)
            # print("\t" * indent + f"vertices layout: {vertices_layout}")

            # normals_layout = bin_buffer.tell(), normals.shape[0]
            # bin_buffer.write(normals)
            # print("\t" * indent + f"normals layout: {normals_layout}")

            # # Indices
            # indices_layout = bin_buffer.tell(), indices.shape[0]
            # bin_buffer.write(indices)

            # bin_data = bin_buffer.getvalue()
            # hash = md5(bin_data).hexdigest()

            # mesh_id = "@mesh-" + str(random.randint(int(1e9), int(1e10 - 1)))
            # mesh = SimMesh(
            #     id=mesh_id,
            #     indicesLayout=indices_layout,
            #     verticesLayout=vertices_layout,
            #     dataHash=hash,
            #     normalsLayout=(0, 0),
            #     uvLayout=(0, 0),
            # )

            indices = mesh_obj.faces.astype(np.int32)
            bin_buffer = io.BytesIO()
            # Vertices
            verts = mesh_obj.vertices.astype(np.float32)
            verts[:, 2] = -verts[:, 2]
            verts = verts.flatten()
            vertices_layout = bin_buffer.tell(), verts.shape[0]
            bin_buffer.write(verts)
            # Normals
            norms = mesh_obj.vertex_normals.astype(np.float32)
            norms[:, 2] = -norms[:, 2]
            norms = norms.flatten()
            normal_layout = bin_buffer.tell(), norms.shape[0]
            bin_buffer.write(norms)
            # Indices
            indices = mesh_obj.faces.astype(np.int32)
            indices = indices[:, [2, 1, 0]]
            indices = indices.flatten()
            indices_layout = bin_buffer.tell(), indices.shape[0]
            bin_buffer.write(indices)
            # Texture coords
            uv_layout = (0, 0)
            if hasattr(mesh_obj.visual, "uv"):
                uvs = mesh_obj.visual.uv.astype(np.float32)
                uvs[:, 1] = 1 - uvs[:, 1]
                uvs = uvs.flatten()
                uv_layout = bin_buffer.tell(), uvs.shape[0]

            bin_data = bin_buffer.getvalue()
            hash = md5(bin_data).hexdigest()

            mesh_id = "@mesh-" + str(random.randint(int(1e9), int(1e10 - 1)))
            mesh = SimMesh(
                id=mesh_id,
                indicesLayout=indices_layout,
                verticesLayout=vertices_layout,
                normalsLayout=normal_layout,
                uvLayout=uv_layout,
                dataHash=hash,
            )

            self.sim_scene.meshes.append(mesh)
            self.sim_scene.raw_data[mesh.dataHash] = bin_data

            #!
            #! do not create new mesh when multiple primitives point to the same prototype
            #!

            sim_mesh = SimVisual(
                type=VisualType.MESH,
                mesh=mesh_id,
                color=[1.0, 1.0, 1.0, 1.0],
                trans=SimTransform(),
            )
            sim_object.visuals.append(sim_mesh)

        # track prims with rigid objects attached
        if (attr := root.GetAttribute("physics:rigidBodyEnabled")) and attr.Get():
            print(f"tracking {sim_object.name}")
            self.tracked_prims.append(
                {"name": sim_object.name, "prim": root, "prim_path": prim_path}
            )

        child: Usd.Prim

        if root.IsInstance():
            proto = root.GetPrototype()
            print("\t" * indent + f"@prototype: {proto.GetPrimPath()}")

            for child in proto.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child, indent=indent + 1, parent_path=prim_path
                ):
                    sim_object.children.append(obj)

        else:
            for child in root.GetChildren():
                if obj := self.parse_prim_tree(
                    root=child, indent=indent + 1, parent_path=prim_path
                ):
                    sim_object.children.append(obj)

        return sim_object

    def get_update(self) -> dict[str, list[float]]:
        import omni
        import omni.usd

        from pxr import Usd, UsdUtils
        from usdrt import Usd as RtUsd
        from usdrt import UsdGeom as RtGeom
        from usdrt import Rt

        def print_state():
            prim = self.rt_stage.GetPrimAtPath("/World/Origin1/Robot/panda_hand")
            print(prim)
            print(prim.GetTypeName())

            prim = Rt.Xformable(prim)
            print(prim.GetWorldPositionAttr().Get())

            rot = prim.GetWorldOrientationAttr().Get()
            print(rot)
            print(
                rot.GetReal(),
                rot.GetImaginary()[0],
                rot.GetImaginary()[1],
                rot.GetImaginary()[2],
            )

        # print_state()

        state = {}

        # for name, trans in self.tracked_obj_trans.items():
        #     pos, rot = trans
        #     state[name] = [-pos[1], pos[2], pos[0], rot[2], -rot[3], -rot[1], rot[0]]

        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

        # print()
        # print(timecode)
        for tracked_prim in self.tracked_prims:
            prim_name = tracked_prim["name"]
            prim_path = tracked_prim["prim_path"]
            prim = tracked_prim["prim"]

            # cur_trans = get_physx_interface().get_rigidbody_transformation(prim_path)
            # print(cur_trans)

            trans_mat = omni.usd.get_world_transform_matrix(prim, timecode)
            # print(f"{prim_name}: {trans_mat}")

            translate = trans_mat.ExtractTranslation()
            translate = [-translate[1], translate[2], translate[0]]

            rot = trans_mat.ExtractRotationQuat()
            imag = rot.GetImaginary()
            rot = [imag[1], -imag[2], -imag[0], rot.GetReal()]

            state[prim_name] = [
                translate[0],
                translate[1],
                translate[2],
                rot[0],
                rot[1],
                rot[2],
                rot[3],
            ]

            rt_prim = self.rt_stage.GetPrimAtPath(prim_path)
            # print(rt_prim)
            # print(rt_prim.GetTypeName())

            rt_prim = Rt.Xformable(rt_prim)
            pos = rt_prim.GetWorldPositionAttr().Get()
            rot = rt_prim.GetWorldOrientationAttr().Get()

            # print(rt_prim.GetWorldPositionAttr().Get())
            # print(rot)
            # print(
            #     rot.GetReal(),
            #     rot.GetImaginary()[0],
            #     rot.GetImaginary()[1],
            #     rot.GetImaginary()[2],
            # )

            state[prim_name] = [
                -pos[1],
                pos[2],
                pos[0],
                rot.GetImaginary()[1],
                -rot.GetImaginary()[2],
                -rot.GetImaginary()[0],
                rot.GetReal(),
            ]

        return state
