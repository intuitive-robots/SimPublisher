from __future__ import annotations
from xml.etree.ElementTree import Element as XMLNode
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import copy
from dataclasses import field
import os
from os.path import join as pjoin
from typing import List, Dict, Callable
import numpy as np
from simpub.loaders.asset_loader import MeshLoader, TextureLoader
from simpub.loaders.json import JsonScene
from simpub.simdata import UnityGameObject, UnityJoint, UnityJointType
from simpub.simdata import UnityMaterial, UnityTransform, UnityVisual
from simpub.simdata import UnityVisualType, UnityScene
from pathlib import Path


class MJCFScene(UnityScene):

    def __repr__(self):
        return f"<MJCFScene {self._path} meshes={len(self.meshes)}>"


class MJCFDefault:
    def __init__(
        self,
        xml: XMLNode = None,
        parent: MJCFDefault = None,
    ) -> None:
        self._dict: Dict[str, Dict[str, str]] = dict()
        self.inherit_fron_parent(xml, parent)

    def import_xml(self, xml: XMLNode) -> None:
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in self._dict.keys():
                self._dict[child.tag] = child.attrib
            else:
                self._dict[child.tag].update(child.attrib)

    def inherit_fron_parent(
        self,
        xml: XMLNode,
        parent: MJCFDefault,
    ) -> None:
        default_dict: Dict[str, Dict[str, str]] = copy.deepcopy(parent._dict)
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in default_dict.keys():
                default_dict[child.tag] = copy.deepcopy(child.attrib)
            else:
                default_dict[child.tag].update(child.attrib)
        self._dict = default_dict

    def update_xml(self, xml: XMLNode) -> None:
        if xml.tag in self._dict:
            attrib = copy.deepcopy(self._dict[xml.tag])
            attrib.update(xml.attrib)
            xml.attrib = attrib


class MJCFTransform:

    RMap: Dict[str, Callable] = {
        "quat": lambda x: MJCFTransform.quat2euler(x),
        "axisangle": lambda x: MJCFTransform.axisangle2euler(x),
        "euler": lambda x: MJCFTransform.euler2euler(x),
        "xyaxes": lambda x: MJCFTransform.xyaxes2euler(x),
        "zaxis": lambda x: MJCFTransform.zaxis2euler(x),
    }

    @classmethod
    def str2array(cls, string, dtype=np.float32, sep=' '):
        return np.fromstring(string, dtype=dtype, sep=sep)

    @classmethod
    def rotation2unity_euler(cls, rotation: Rotation):
        return rotation.as_euler("xyz", degrees=True)

    @classmethod
    def quat2euler(cls, quat: np.ndarray):
        quat = np.asarray(quat, dtype=np.float32)
        assert len(quat) == 4, "Quaternion must have four components."
        # Mujoco use wxyz format and Unity uses xyzw format
        w, x, y, z = quat
        quat = np.array([x, y, z, w], dtype=np.float64)
        return cls.rotation2unity_euler(Rotation.from_quat(quat))

    @classmethod
    def axisangle2euler(cls, axisangle: np.ndarray, use_degree=True):
        assert len(axisangle) == 4, (
            "axisangle must contain four values (x, y, z, a)."
        )
        # Extract the axis (x, y, z) and the angle a
        axis = axisangle[:3]
        angle = axisangle[3]
        axis = axis / np.linalg.norm(axis)
        if use_degree:
            angle = np.deg2rad(angle)
        rotation = Rotation.from_rotvec(angle * axis)
        return cls.rotation2unity_euler(rotation)

    @classmethod
    def euler2euler(cls, euler: np.ndarray, degree: str = True):
        assert len(euler) == 3, "euler must contain three values (x, y, z)."
        # Convert the Euler angles to radians if necessary
        if not degree:
            euler = np.rad2deg(euler)
        return euler

    @classmethod
    def xyaxes2euler(cls, xyaxes: np.ndarray):
        assert len(xyaxes) == 6, (
            "xyaxes must contain six values (x1, y1, z1, x2, y2, z2)."
        )
        x = xyaxes[:3]
        y = xyaxes[3:]
        z = np.cross(x, y)
        rotation_matrix = np.array([x, y, z]).T
        rotation = Rotation.from_matrix(rotation_matrix)
        return cls.rotation2unity_euler(rotation)

    @classmethod
    def zaxis2euler(cls, zaxis: np.ndarray):
        assert len(zaxis) == 3, "zaxis must contain three values (x, y, z)."
        # Create the rotation object from the z-axis
        rotation = Rotation.from_rotvec(np.pi, zaxis)
        return cls.rotation2unity_euler(rotation)

    @staticmethod
    def mj2unity(pos):
        return np.array([-pos[1], pos[2], -pos[0]])


class MJCFUnityPrimitive:
    PrimitiveType: Dict[str, UnityVisualType] = {
        "plane": UnityVisualType.PLANE,
        "sphere": UnityVisualType.SPHERE,
        "capsule": UnityVisualType.CAPSULE,
        "ellipsoid": UnityVisualType.CAPSULE,
        "cylinder": UnityVisualType.CYLINDER,
        "box": UnityVisualType.CUBE,
        "mesh": UnityVisualType.MESH
    }

    ScaleMap: Dict[str, Callable] = {
        "plane": lambda x: MJCFUnityPrimitive.plane2unity_scale(x),
        "box": lambda x: MJCFUnityPrimitive.box2unity_scale(x),
        "sphere": lambda x: MJCFUnityPrimitive.sphere2unity_scale(x),
        "cylinder": lambda x: MJCFUnityPrimitive.cylinder2unity_scale(x),
        "capsule": lambda x: MJCFUnityPrimitive.capsule2unity_scale(x),
        "ellipsoid": lambda x: MJCFUnityPrimitive.capsule2unity_scale(x),
    }

    @classmethod
    def scale2unity(cls, scale, visual_type: str):
        return cls.ScaleMap[visual_type](scale)

    @staticmethod
    def plane2unity_scale(scale):
        return np.abs(np.array([
            (scale[0] or 100.0) * 2,
            0.001,
            (scale[1] or 100) * 2
        ]))

    @staticmethod
    def box2unity_scale(scale):
        return np.abs(scale * 2)

    @staticmethod
    def sphere2unity_scale(scale):
        return np.abs(np.array([scale[0] * 2] * 3))

    @staticmethod
    def cylinder2unity_scale(scale):
        if len(scale) == 3:
            return np.abs(np.array([scale[0], scale[1], scale[0]]))
        else:
            return np.abs(np.array([scale[0] * 2, scale[1], scale[0] * 2]))

    @staticmethod
    def capsule2unity_scale(scale):
        assert len(scale) == 3, "Only support scale with three components."
        return np.abs(np.array([scale[0], scale[1], scale[0]]))


class MJCFParser:
    def __init__(
        self,
        file_path: str,
        no_tracked_joints: List[str] = field(default_factory=list[str]),
        no_rendered_object: List[str] = field(default_factory=list[str]),
    ):
        self._xml_path = os.path.abspath(file_path)
        self._path = os.path.join(self._xml_path, "..")
        self.no_tracked_joints = no_tracked_joints
        self.no_rendered_object = no_rendered_object

    def parse(
        self, no_render_object_list: List[str] = field(default_factory=list),
    ) -> MJCFScene:
        self.no_render_object_list = no_render_object_list
        raw_xml = self.get_root_from_xml_file(self._path)
        return self._pre_parse(raw_xml)

    def get_root_from_xml_file(self, xml_path: str) -> XMLNode:
        xml_path = os.path.abspath(xml_path)
        assert os.path.exists(xml_path), (f"File '{xml_path}' does not exist.")
        tree_xml = ET.parse(xml_path)
        return tree_xml.getroot()

    def _parse_xml(self, raw_xml: XMLNode) -> MJCFScene:
        mj_scene = MJCFScene()
        xml = self._merge_includes(raw_xml)
        self._load_compiler(xml)
        self._load_defaults(xml)
        self._load_assets(mj_scene)
        self._load_worldbody(raw_xml, mj_scene)
        return mj_scene

    def _merge_includes(self, root_xml: XMLNode) -> XMLNode:
        for child in root_xml:
            if child.tag != "include":
                self._merge_includes(child)
                continue
            sub_xml_path = os.path.join(self._path, "..", child.attrib["file"])
            sub_xml_root = self.get_root_from_xml_file(sub_xml_path)
            root_xml.extend(sub_xml_root)
            root_xml.remove(child)
        return root_xml

    def _load_compiler(self, xml: XMLNode) -> None:
        for compiler in xml.findall("./compiler"):
            self._use_degree = (
                True if compiler.get("angle", "degree") == "degree" else False
            )
            self._eulerseq = compiler.get("eulerseq", "xyz")
            self._assetdir = pjoin(self._path, compiler.get("assetdir", ""))
            self._meshdir = pjoin(self._path, compiler.get("meshdir", ""))
            self._texturedir = pjoin(self._path, compiler.get("texturedir", ""))

    def _load_defaults(self, root_xml: XMLNode) -> None:
        default_dict: Dict[str, MJCFDefault] = dict()
        default_dict["main"] = MJCFDefault()
        # only start _loading default tags under the mujoco tag
        for default_child_xml in root_xml.findall("default"):
            self._parse_default(default_child_xml, default_dict)
        # replace the class attribute with the default values
        self._import_default(root_xml, default_dict)

    def _parse_default(
        self,
        default_xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent: MJCFDefault = None
    ) -> None:
        if parent is None:
            default = default_dict["main"]
            default.import_xml(default_xml)
        else:
            default = MJCFDefault(default_xml, parent)
            default_dict[default.class_name] = default
        for default_child_xml in default_xml.findall("default"):
            self._parse_default(default_child_xml, default_dict, default)

    def _import_default(
        self,
        xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent: MJCFDefault = None
    ) -> None:
        if xml.tag == "default":
            return
        if parent is None:
            parent = MJCFDefault.Top
        default = (
            default_dict[xml.attrib["class"]]
            if "class" in xml.attrib.keys()
            else parent
        )
        default.update_xml(xml)
        parent = (
            default_dict[xml.attrib["childclass"]]
            if "childclass" in xml.attrib
            else parent
        )
        for child in xml:
            self._import_default(child, parent)

    def _load_assets(self, xml: XMLNode, mj_scene: MJCFScene) -> None:

        for asset in xml.findall("./asset"):
            if asset.tag == "mesh":
                # REVIEW: the link you are using is for latest xml, please check the mujoco210 version
                # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
                asset_file = pjoin(self._meshdir, asset.attrib["file"])
                scale = np.fromstring(
                    asset.get("scale", "1 1 1"), dtype=np.float32, sep=" "
                )
                mesh, bin_data = MeshLoader.from_file(
                    asset_file, asset.get("name", asset_file.stem), scale
                )
                mj_scene.meshes.append(mesh)
                mj_scene._raw_data[mesh.dataID] = bin_data

            elif asset.tag == "material":
                # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material
                color = np.fromstring(
                    asset.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' '
                )
                material = UnityMaterial(
                    tag=asset.get("name") or asset.get("type"),
                    color=color,
                    emissionColor=float(asset.get("emission", "0.0")) * color,
                    specular=float(asset.get("specular", "0.5")),
                    shininess=float(asset.get("shininess", "0.5")),
                    reflectance=float(asset.get("reflectance", "0.0"))
                )
                material.texture = asset.get("texture", None)
                material.texsize = np.fromstring(
                    asset.get("texrepeat", "1 1"), dtype=np.int32, sep=' '
                )
                mj_scene.materials.append(material)
            elif asset.tag == "texture":
                # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture
                name = asset.get("name") or asset.get("type")
                builtin = asset.get("builtin", "none")
                tint = np.fromstring(
                    asset.get("rgb1", "1 1 1"), dtype=np.float32, sep=' '
                )
                if builtin != "none":
                    texture, bin_data = TextureLoader.fromBuiltin(
                        name, builtin, tint
                    )
                else:
                    asset_file = pjoin(self._texturedir, asset.attrib["file"])
                    byte_data = Path(asset_file, "rb").read_bytes()
                    texture, bin_data = TextureLoader.from_bytes(
                        name, byte_data, asset.get("type", "cube"), tint
                    )
                mj_scene.textures.append(texture)
                mj_scene._raw_data[texture.dataID] = bin_data
            else:
                raise RuntimeError("Invalid asset", asset.tag)

    def _load_worldbody(self, xml: XMLNode) -> None:
        for worldbody in xml.findall("./worldbody"):
            self.worldbody = self._load_body(worldbody)

    def get_rotation_from_xml(self, obj_xml: XMLNode) -> np.ndarray:
        result: np.ndarray = np.array([0, 0, 0])
        for key in MJCFTransform.RMap.keys():
            if key in obj_xml.attrib:
                result = MJCFTransform.RMap[key](
                    MJCFTransform.str2array(obj_xml.get(key))
                )
                break
        return result

    def _load_visual(self, visual: XMLNode, group: int) -> UnityVisual:
        transform = UnityTransform(
            position=MJCFTransform.mj2unity(
                MJCFTransform.str2array(visual.get("pos", "0 0 0"))
            ),
            rotation=self.get_rotation_from_xml(visual),
            scale=np.abs(MJCFTransform.str2array(visual.get("size", "1 1 1"))),
        )

        visual_type = visual.get("type")
        if (visual_type not in MJCFUnityPrimitive.PrimitiveType or
                visual_type is None):
            print(f"WARNING: Unknown visual type {visual_type}")
            return UnityVisual(
                type="sphere",
                transform=transform,
            )

        return UnityVisual(
            type=type,
            transform=transform,
            mesh=visual.get("mesh"),
            material=visual.get("material"),
            color=np.fromstring(visual.get("rgba", "1 1 1 1"), dtype=np.float32, sep=' ')
        )

    # TODO: Check the if joint should be free joint or still we need differnt joint types
    def _load_joint(self, joint: XMLNode, parent_name: str):
        range = np.fromstring(joint.get("range", "1000000 100000"), dtype=np.float32, sep=' ')
        if self._use_degree:
            range = np.degrees(range)
        return UnityJoint(
            name=joint.get("name", parent_name),
            axis=MJCFTransform.mj2unity(
                MJCFTransform.str2array(joint.get("axis", "1 0 0"))
            ),
            minrot=float(range[0]),
            maxrot=float(range[1]),
            transform=UnityTransform(
                position=mj2pos(np.fromstring(joint.get("pos", "0 0 0"), dtype=np.float32, sep=' '))
            ),
            type=UnityJointType(joint.get("type", "hinge").upper()),
        )

    def _load_body(self, body: XMLNode) -> UnityGameObject:
        name = body.get("name", "worldbody")
        if name in self.no_rendered_object:
            return None

        transform = UnityTransform(
            position=MJCFTransform.mj2unity(
                MJCFTransform.str2array(body.get("pos", "0 0 0"))
            ),
            rotation=self.get_rotation_from_xml(body)
        )

        # group = max(set(int(val) for geom in body.findall("./geom") if (val := geom.get("group") is not None)) or { 0 })

        joint: UnityJoint
        if body.find("./freejoint") is not None:
            joint = UnityJoint(type=UnityJointType.FREE, name=name)
        for joint_xml in body.findall("./joint"):
            if joint_xml.get("name") in self.no_tracked_joints:
                continue
            elif joint_xml.get("type") == "free":
                joint = UnityJoint(type=UnityJointType.FREE, name=name)
            else:
                joint = [UnityJoint(name)]
        # the first joint sets the frame of reference for the body
        joint.transform = joint.transform + transform

        visuals: List[UnityVisual] = list()
        for geom in body.findall("geom") + body.findall("site"):
            visual = self._load_visual(geom, group)
            if visual is not None:
                visuals.append(visual)

        return UnityGameObject(
            name=name,
            joint=joint,
            visuals=visuals,
            children=[self._load_body(child) for child in body.findall("body")],
        )


if __name__ == "__main__":
    file = "scenes/anybotics_anymal_b/scene.xml"

    print("loading", file)

    mj = MJCFScene.from_file(file)
    print(mj)

    scene = mj.to_scene()
    print(JsonScene.to_string(scene))
