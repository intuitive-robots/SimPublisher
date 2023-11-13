from alr_sim.core.sim_object import SimObject
from alr_sim.core.Scene import Scene
from alr_sim.sims.universal_sim import PrimitiveObjects

from sim_pub.base import ObjectPublisherBase, SimPubDataBlock
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
        sim_obj: SimObject,
        scene: Scene,
        **kwargs,
    ) -> None:
        super().__init__(id)
        self.sim_obj = sim_obj
        self.scene = scene
        self.param = SimPubDataBlock()
        self.state = SimPubDataBlock()

        str_dict = self.param["str_dict"]
        list_dict = self.param["list_dict"]
        bool_dict = self.param["bool_dict"]
        str_dict["id"] = sim_obj.name
        for k, v in kwargs.items():
            if type(v) is str:
                str_dict[k] = v
            elif type(v) is list[float]:
                list_dict[k] = v
            elif type(v) is bool:
                bool_dict[k] = v
            else:
                raise TypeError

    def get_obj_param_dict(self) -> SimPubDataBlock:
        return self.param

    def get_obj_state_dict(self) -> SimPubDataBlock:
        return self.state

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
        self.get_obj_pos_fct = self.scene.get_obj_pos
        self.get_obj_quat_fct = self.scene.get_obj_quat
        
    def get_obj_state_dict(self) -> SimPubDataBlock:
        return {
            "Header": "sim_state",
            "data": {
                "pos": list(mj2unity_pos(self.get_obj_pos_fct(self.obj))),
                "rot": list(mj2unity_quat(self.get_obj_quat_fct(self.obj))),
            }
        }

class SFPrimitiveObjectPublisher(SFRigidBodyPublisher):
    
    def __init__(
        self, sim_obj: PrimitiveObjects.PrimitiveObject, 
        scene: Scene, 
    ) -> None:
        super().__init__(
            sim_obj, 
            scene, 
        )
        list_dict = self.param["list_dict"]
        if type(sim_obj) is PrimitiveObjects.Box:
            self.param["strings"] = "Box"
            list_dict["size"] = [sim_obj.size[1] * 2, sim_obj.size[2] * 2, sim_obj.size[0] * 2]
        elif type(sim_obj) is PrimitiveObjects.Sphere:
            self.param["strings"] = "Sphere"
            list_dict["size"] = [sim_obj.size[1], sim_obj.size[2], sim_obj.size[0]]
        elif type(sim_obj) is PrimitiveObjects.Cylinder:
            self.param["strings"] = "Cylinder"
            list_dict["size"] = [sim_obj.size[1], sim_obj.size[2] * 2, sim_obj.size[0]]
        else:
            raise TypeError


class SFYCBObjectPublisher(SFRigidBodyPublisher):
    
    def __init__(
        self, sim_obj: SimObject, 
        scene: Scene, 
        size: list[float] = [1, 1, 1], 
        rgba: list[float] = [-1, -1, -1, 1], 
        static: bool = False, 
        interactable: bool = False
    ) -> None:
        super().__init__(sim_obj, scene, size, rgba, static, interactable)
        self.param[]

class SFPandaPublisher(SFObjectPublisher):
    pass
