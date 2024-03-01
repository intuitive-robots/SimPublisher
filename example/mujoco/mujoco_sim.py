import mujoco_py
import os

from sim_pub.model_loader import SceneLoader

# Load the model from an XML file
model_path = os.path.join(os.path.dirname(__file__), "pendulum.xml")
model = mujoco_py.load_model_from_path(model_path)

scene_loader = SceneLoader()
scene_loader.include_mjcf_file(model_path)
print(scene_loader.generate_model_dict())

# sim = mujoco_py.MjSim(model)

# # Create a viewer to visualize the simulation
# viewer = mujoco_py.MjViewer(sim)

# # Simulate for 100 steps
# for _ in range(100000):
#     sim.step()
#     viewer.render()

print("Simulation done")
