from pathlib import Path
import unittest

import numpy as np

from simpub.loaders.mjcf_parser import MJCFScene
from simpub.simdata import SimVisualType


class TestMyFunction(unittest.TestCase):
    # This function tests the correct behaviour of childclass and class attributes
    def test_childclass_behaviour(self):
        xml = """
              <mujoco>
                  <default class="main">
                      <geom rgba="1 0 0 1"/>
                      <default class="sub">
                          <geom rgba="0 1 0 1"/>
                      </default>
                  </default>

                  <worldbody>
                      <geom type="box"/>
                      <body childclass="sub">
                          <geom type="ellipsoid"/>
                          <geom type="sphere" rgba="0 0 1 1"/>
                          <geom type="cylinder" class="main"/>
                      </body>
                  </worldbody>
              </mujoco>
              """
        scene = MJCFScene.from_string(xml, Path("."))

        box = scene.worldbody.visuals[0]
        np.testing.assert_array_equal(box.color, np.array([1, 0, 0, 1]))
        
        ellipsoid = scene.worldbody.bodies[0].visuals[0]
        self.assertEqual(ellipsoid.type, SimVisualType.CAPSULE)
        np.testing.assert_array_equal(ellipsoid.color, np.array([0, 1, 0, 1]))

        
        sphere = scene.worldbody.bodies[0].visuals[1]
        self.assertEqual(sphere.type, SimVisualType.SPHERE)
        np.testing.assert_array_equal(sphere.color, np.array([0, 0, 1, 1]))


        cylinder = scene.worldbody.bodies[0].visuals[2]
        self.assertEqual(cylinder.type, SimVisualType.CYLINDER)
        np.testing.assert_array_equal(cylinder.color, np.array([1, 0, 0, 1]))

    def test_compiler_settings(self):
        xml = """
            <mujoco>
             <compiler angle="radian" meshdir="assets" texturedir="textures" autolimits="true" />
            </mujoco>
            """ 

        scene = MJCFScene.from_string(xml, Path("."))

        self.assertEqual(scene.angle, "radian")
        self.assertEqual(scene.meshdir, Path("assets"))
        self.assertEqual(scene.texturedir, Path("textures"))
    

if __name__ == '__main__':
    unittest.main()