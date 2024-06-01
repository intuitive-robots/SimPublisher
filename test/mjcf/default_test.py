from pathlib import Path
import unittest

import numpy as np

from simpub.loaders.mjcf_parser import MJCFFile
from simpub.simdata import SimVisualType


class TestMyFunction(unittest.TestCase):
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
        scene = MJCFFile.from_string(xml, Path(".")).to_scene()

        box = scene.worldbody.visuals[0]
        np.testing.assert_array_equal(box.color, np.array([1, 0, 0, 1]))
        
        ellipsoid = scene.worldbody.joints[0].body.visuals[0]
        self.assertEqual(ellipsoid.type, SimVisualType.CAPSULE)
        print(ellipsoid.color)
        np.testing.assert_array_equal(ellipsoid.color, np.array([0, 1, 0, 1]))

        
        sphere = scene.worldbody.joints[0].body.visuals[1]
        self.assertEqual(sphere.type, SimVisualType.SPHERE)
        np.testing.assert_array_equal(sphere.color, np.array([0, 0, 1, 1]))

        
        
        cylinder = scene.worldbody.joints[0].body.visuals[2]
        self.assertEqual(cylinder.type, SimVisualType.CYLINDER)
        np.testing.assert_array_equal(cylinder.color, np.array([1, 0, 0, 1]))


if __name__ == '__main__':
    unittest.main()