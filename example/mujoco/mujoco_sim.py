import mujoco_py
import os

import sys
import os

from sim_pub.model_loader import SceneLoader
from sim_pub.mujoco.mujoco_streamer import MujocoStreamer

# Load the model from an XML file
model_path = os.path.join(os.path.dirname(__file__), "pendulum.xml")
# model = mujoco_py.load_model_from_path(model_path)

# scene_loader = SceneLoader()
# scene_loader.include_mjcf_file(model_path)

streamer = MujocoStreamer(model_path)

streamer.start_server_thread(block=True)

# sim = mujoco_py.MjSim(model)

# # Create a viewer to visualize the simulation
# viewer = mujoco_py.MjViewer(sim)

# # Simulate for 100 steps
# for _ in range(100000):
#     sim.step()
#     viewer.render()

print("Simulation done")
