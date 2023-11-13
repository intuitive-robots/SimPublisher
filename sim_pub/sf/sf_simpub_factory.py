from alr_sim.core.sim_object import SimObject
from alr_sim.sims.universal_sim import PrimitiveObjects

from sf import SFObjectPublisher


class SFSimPubFactory:

    def create_publisher(self, sim_obj: SimObject) -> SFObjectPublisher:
        
        if type(sim_obj) is PrimitiveObjects.Box:
            obj_size = [sim_obj.size[1] * 2, sim_obj.size[2] * 2, sim_obj.size[0] * 2]
        elif type(sim_obj) is PrimitiveObjects.Sphere:
            obj_size = [sim_obj.size[1], sim_obj.size[2], sim_obj.size[0]]
        elif type(sim_obj) is PrimitiveObjects.Cylinder:
            obj_size = [sim_obj.size[1], sim_obj.size[2] * 2, sim_obj.size[0]]
        else:
            obj_size = [1, 1, 1]
        return 