from simpub.mjcf.mj_publisher import MjPublisher
from alr_sim.sims.mj_beta import MjScene
from .sf_parser import SFParser

import json


class SFPublisher(MjPublisher):

    def __init__(self, sf_mj_sim: MjScene) -> None:
        self.parser = SFParser(sf_mj_sim)
        self.unity_scene_data = self.parser.parse()
        self.scene_message = self.unity_scene_data.to_string()
        self.mj_data: MjScene = sf_mj_sim.data
        self.mj_model: MjScene = sf_mj_sim.model
        self.config_zmq()
        self.track_joints_from_unity_scene(self.unity_scene_data)

        def print_hierarchy(node, indent=0):
            print('  ' * indent + f"Name: {node['name']}")
            print('  ' * indent + f"  Transform: {node['trans']}")
            print('  ' * indent + f"  Joint: {node['joint']}")
            print('  ' * indent + f"  Visuals: {node['visuals']}")
            for child in node.get('children', []):
                print_hierarchy(child, indent + 1)
        print_hierarchy(json.loads(self.scene_message)["root"])