import abc
from typing import Union

from alr_sim.core.sim_object import SimObject
from alr_sim.core.Robots import RobotBase
from alr_sim.core.Scene import Scene
from alr_sim.sims.universal_sim import PrimitiveObjects
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import YCBMujocoObject

from sim_pub.base import ObjectPublisherBase, SimPubDataBlock, SimPubData
from sim_pub.primitive import SimStreamer
from sim_pub.utils import *
from sim_pub.geometry import *

from sf import SFObjectPublisher

class SFObjectPublisher(ObjectPublisherBase):
    """_summary_

    Args:
        ObjectPublisherBase (_type_): _description_
    """

    def __init__(
        self,
        sim_obj: Union[SimObject, RobotBase],
        scene: Scene,
        **kwargs,
    ) -> None:
        super().__init__(id)
        self.sim_obj = sim_obj
        self.scene = scene
        param = SimPubDataBlock()
        for k, v in kwargs.items():
            param.add_value(k, v)
        self.param = param

    def update_obj_param_dict(self, data: SimPubData) -> None:
        data[self.id] = self.param

    @abc.abstractmethod
    def update_obj_state_dict(self, data: SimPubData) -> None:
        raise NotImplemented


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
        scene: Scene,
        **kwargs,
    ) -> None:

        super().__init__(id, sim_obj, scene, **kwargs)


    def update_obj_state_dict(self, data: SimPubData) -> None:
        state = SimPubDataBlock()
        state["pos"] = list(mj2unity_pos(self.scene.get_obj_pos(self.sim_obj)))
        state["quat"] = list(mj2unity_pos(self.scene.get_obj_quat(self.sim_obj)))
        data[self.id] = state


class SFPrimitiveObjectPublisher(SFRigidBodyPublisher):
    
    def __init__(
        self, sim_obj: PrimitiveObjects.PrimitiveObject, 
        scene: Scene,
        **kwargs,
    ) -> None:
        super().__init__(sim_obj, scene, **kwargs)
        size_list: List[float] = sim_obj.size
        if type(sim_obj) is PrimitiveObjects.Box:
            self.param.add_str("type", "Box")
            self.param.add_str("size", [size_list[1] * 2, size_list[2] * 2, size_list[0] * 2])
        elif type(sim_obj) is PrimitiveObjects.Sphere:
            self.param.add_str("type", "Sphere")
            self.param.add_str("size", [size_list[1], size_list[2], size_list[0]])
        elif type(sim_obj) is PrimitiveObjects.Cylinder:
            self.param.add_str("type", "Cylinder")
            self.param.add_str("size", [size_list[1], size_list[2] * 2, size_list[0]])
        else:
            raise TypeError


class SFYCBObjectPublisher(SFRigidBodyPublisher):
    
    def __init__(
        self, sim_obj: YCBMujocoObject, 
        scene: Scene, 
        **kwargs,
    ) -> None:
        super().__init__(sim_obj, scene, **kwargs)
        self.param["rot_offset"] = [-90.0, 0.0, 90.0]

class SFPandaPublisher(SFObjectPublisher):
    
    def __init__(
        self, 
        robot: RobotBase, 
        scene: Scene, 
        **kwargs
    ) -> None:
        super().__init__(robot, scene, **kwargs)


    def update_obj_state_dict(self, data: SimPubData) -> None:
        state = SimPubDataBlock()
        joints = list(self.sim_obj.current_j_pos)
        joints.extend([self.sim_obj.gripper_width / 2, self.sim_obj.gripper_width / 2])
        state.add_str("joints", joints)
        data[self.id] = state
