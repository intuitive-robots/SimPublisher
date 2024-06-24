from pathlib import Path
import unittest

import numpy as np

from simpub.mjcf.mjcf_parser import MJCFParser
from simpub.unity_data import UnityJointType, UnityVisualType


class TestMJCFParserFunction(unittest.TestCase):
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.parser = MJCFParser("test/mjcf/test_model/childclass_example.xml")
        self.scene = self.parser.parse()

    # This function tests the behaviour of childclass and class attributes
    def test_childclass_behaviour(self):

        box = self.scene.objects[0].visuals[0]
        np.testing.assert_array_equal(box.color, [1, 0, 0, 1])

        ellipsoid = self.scene.objects[1].visuals[0]
        self.assertEqual(ellipsoid.type, UnityVisualType.CAPSULE)
        np.testing.assert_array_equal(ellipsoid.color, [0, 1, 0, 1])

        sphere = self.scene.objects[1].visuals[1]
        self.assertEqual(sphere.type, UnityVisualType.SPHERE)
        np.testing.assert_array_equal(sphere.color, [0, 0, 1, 1])

        cylinder = self.scene.objects[1].visuals[2]
        self.assertEqual(cylinder.type, UnityVisualType.CYLINDER)
        np.testing.assert_array_equal(cylinder.color, [1, 0, 0, 1])

    def test_compiler_settings(self):
        self.assertEqual(self.parser._use_degree, False)
        # TODO: More tests about whether settings are configed correctly
        # self.assertEqual(self.parser._meshdir, Path("assets"))
        # self.assertEqual(self.parser._texturedir, Path("textures"))

    # def test_body_layout(self):
    #     # This function tests the correct behaviour of worldbody parsing
    #     xml = """
    #           <mujoco>
    #               <worldbody>
    #                   <geom type="box"/>
    #                   <body name="first" pos="1 1 1" euler="0.9 90. 43">
    #                         <geom type="sphere" rgba="0 0 1 1"/>
    #                         <geom type="box" rgba="0 0 1 1"/>
    #                         <geom type="box" rgba="0 0 1 1"/>
    #                         <joint type="hinge" axis="1 0 0"/>
    #                   </body>
    #                   <body name="second/third">
    #                     <freejoint/>
    #                   </body>
    #               </worldbody>
    #           </mujoco>
    #           """
    
    #     scene = MJCFScene.from_string(xml, Path("."))

    #     self.assertEqual(scene.worldbody.bodies[0].name, "first")
    #     self.assertEqual(scene.worldbody.bodies[1].name, "second/third")


    #     self.assertEqual(scene.worldbody.bodies[1].joints[0].type, UnityJointType.FREE)

    #     self.assertEqual(scene.worldbody.visuals[0].type, UnityVisualType.BOX)

        
    #     self.assertEqual(scene.worldbody.bodies[0].visuals[0].type, UnityVisualType.SPHERE)
    #     self.assertEqual(scene.worldbody.bodies[0].visuals[1].type, UnityVisualType.BOX)
    #     self.assertEqual(scene.worldbody.bodies[0].visuals[2].type, UnityVisualType.BOX)

        
    #     self.assertEqual(scene.worldbody.bodies[0].joints[0].type, UnityJointType.HINGE)
    #     self.assertListEqual(scene.worldbody.bodies[0].joints[0].axis.tolist(), [0, 0, -1])


if __name__ == '__main__':
    unittest.main()
