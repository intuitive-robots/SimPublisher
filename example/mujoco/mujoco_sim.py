import mujoco
import os

import os

# from sim_pub.model_loader import MJCFImporter

# Load the model from an XML file
model_path = os.path.join(os.path.dirname(__file__), "pendulum.xml")
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)


# scene_importer = MJCFImporter()
# scene_importer.include_xml_file(model_path)

# streamer = MujocoStreamer(model_path)

# streamer.start_server_thread(block=True)
# streamer.start_server_thread(block=False)


sim = mujoco.MjSim(model)

# # Create a viewer to visualize the simulation
viewer = mujoco.MjViewer(sim)

# Simulate for 100 steps
for _ in range(100000):
    sim.step()
    viewer.render()

print("Simulation done")
