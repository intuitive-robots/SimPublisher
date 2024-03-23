import abc
from os import name
from typing import List, Dict, Union

from alr_sim.core.sim_object import SimObject
from alr_sim.core.Robots import RobotBase
from alr_sim.sims.mj_beta import MjScene, MjRobot
from alr_sim.sims.universal_sim import PrimitiveObjects
from alr_sim.sims.universal_sim.PrimitiveObjects import PrimitiveObject
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject
from alr_sim.utils.sim_path import sim_framework_path

from sim_pub.server.base import ObjectPublisherBase, SimPubDataBlock, SimPubData
from sim_pub.server.ws.ws_server import SimStreamer
from sim_pub.sfmj.geometry import *


class SFObjectPublisher(ObjectPublisherBase):
    """_summary_

    Args:
        ObjectPublisherBase (_type_): _description_
    """

    def __init__(
        self,
        name: str,
        scene: MjScene,
        sim_obj: Union[SimObject, RobotBase] = None,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.scene = scene
        self.param_dict: Dict[str, Union[str, List[float], bool]] = {
            k: v for k, v in kwargs.items()
        }
        self.set_up_default_param()
        if sim_obj is None:
            self.create_sim_obj()
        else:
            self.sim_obj = sim_obj

    def update_param_from_dict(**kwargs):
        pass

    @abc.abstractmethod
    def set_up_default_param(self):
        raise NotImplemented
    

    def update_obj_param_dict(self, data: SimPubData) -> None:
        param = SimPubDataBlock()
        for k, v in self.param_dict:
            param.add_value(k, v)
        data[self.id] = param


class SFRigidBodyPublisher(SFObjectPublisher):
    """_summary_

            obj_type: str,
        init_pos: list[float],
        init_quat: list[float],
        size: list[float] = [1, 1, 1],
        rgba: list[float] = [-1, -1, -1, 1],
        static: bool = False,
        interactable: bool = False,
    Args:
        ObjectPublisherBase (_type_): _description_
    """
    def __init__(
        self, 
        sim_obj: SimObject, 
        scene: MjScene,
        **kwargs,
    ) -> None:

        super().__init__(sim_obj.name, sim_obj, scene, **kwargs)

    def full_fill_param_dict(self):
        param = self.param_dict
        param.setdefault("static", False)
        param.setdefault("visual_only", False)
        param.setdefault("interactable", False)

    def update_obj_state_dict(self, data: SimPubData) -> None:
        state = SimPubDataBlock()
        state["pos"] = list(mj2unity_pos(self.scene.get_obj_pos(self.sim_obj)))
        state["quat"] = list(mj2unity_pos(self.scene.get_obj_quat(self.sim_obj)))
        data[self.id] = state


class SFPrimitiveObjectPublisher(SFRigidBodyPublisher):
    
    def __init__(
        self, sim_obj: PrimitiveObject, 
        scene: MjScene,
        **kwargs,
    ) -> None:
        super().__init__(sim_obj, scene, **kwargs)
        self.update_obj_unity_size()

    def create_primitive_obj_from_type(self, primitive_type: type) -> None:
        self.sim_obj = primitive_type(
            name = self.param_dict["obj_name"],
            init_pos = self.param_dict["init_pos"],
            init_quat = self.param_dict["init_quat"],
            mass = 0.1,
            size = self.param_dict["mj_size"],
            rgba = self.param_dict["rgba"],
            static = self.param_dict["static"],
            visual_only = self.param_dict["visual_only"],
        )


class SFPrimitiveBoxPubliser(SFPrimitiveObjectPublisher):

    def __init__(self, sim_obj: PrimitiveObject, scene: MjScene, **kwargs) -> None:
        super().__init__(sim_obj, scene, **kwargs)

    def full_fill_param_dict(self):
        mj_size = self.param_dict["mj_size"]
        self.param_dict["unity_size"] = [mj_size[1] * 2, mj_size[2] * 2, mj_size[0] * 2]
        super().full_fill_param_dict()

    def create_sim_obj(self) -> None:
        self.create_primitive_obj_from_type(PrimitiveObjects.Box)


class SFPrimitiveSpherePubliser(SFPrimitiveObjectPublisher):

    def __init__(self, sim_obj: PrimitiveObject, scene: MjScene, **kwargs) -> None:
        super().__init__(sim_obj, scene, **kwargs)

    def full_fill_param_dict(self):
        mj_size = self.param_dict["mj_size"]
        self.param_dict["unity_size"] = [mj_size[1], mj_size[2], mj_size[0]]
        super().full_fill_param_dict()

    def create_sim_obj(self) -> None:
        self.create_primitive_obj_from_type(PrimitiveObjects.Sphere)


class SFPrimitiveCylinderPubliser(SFPrimitiveObjectPublisher):

    def __init__(self, sim_obj: PrimitiveObject, scene: MjScene, **kwargs) -> None:
        super().__init__(sim_obj, scene, **kwargs)

    def full_fill_param_dict(self):
        mj_size = self.param_dict["mj_size"]
        self.param_dict["unity_size"] = [mj_size[1], mj_size[2] * 2, mj_size[0]]
        super().full_fill_param_dict()

    def create_sim_obj(self) -> None:
        self.create_primitive_obj_from_type(PrimitiveObjects.Cylinder)


class SFYCBObjectPublisher(SFRigidBodyPublisher):
    _ycb_path = "../SFModels/YCB/models/ycb"
    def __init__(
        self, sim_obj: YCBMujocoObject, 
        scene: MjScene, 
        **kwargs,
    ) -> None:
        super().__init__(sim_obj, scene, **kwargs)
        self.param.add_float_list([-90.0, 0.0, 90.0])

    def create_sim_obj(self) -> None:
        self.sim_obj = YCBMujocoObject(
            ycb_base_folder=sim_framework_path("../SF-ObjectDataset/YCB/"),
            object_id = self.param_dict["obj_type"],
            object_name = self.param_dict["obj_name"],
            pos = self.param_dict["init_pos"],
            quat = self.param_dict["init_quat"],
            static = self.param_dict["static"],
            alpha = self.param_dict["riga"][-1],
            visual_only = self.param_dict["visual_only"],
        )

class SFRobotPubliser(SFObjectPublisher):
    
    def __init__(self, scene: MjScene, sim_obj: RobotBase = None, **kwargs) -> None:
        super().__init__(scene, sim_obj, **kwargs)

class SFPandaPublisher(SFRobotPubliser):
    
    def __init__(
        self, 
        robot: MjRobot, 
        scene: MjScene, 
        **kwargs
    ) -> None:
        super().__init__(robot, scene, **kwargs)

    def update_obj_state_dict(self, data: SimPubData) -> None:
        state = SimPubDataBlock()
        joints = list(self.sim_obj.current_j_pos)
        joints.extend([self.sim_obj.gripper_width / 2, self.sim_obj.gripper_width / 2])
        state.add_str("joints", joints)
        data[self.id] = state

    def create_sim_obj(self) -> None:
        return super().create_sim_obj()