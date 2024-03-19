import mujoco
import os

from sim_pub.model_loader import MJCFLoader
from sim_pub.mujoco.mujoco_streamer import MujocoStreamer

# Load the model from an XML file
model_path = os.path.join(os.path.dirname(__file__), "../../models/static_scene/static_scene.xml")
model_path = os.path.abspath(model_path)
# # Create a viewer to visualize the simulation
# model = mujoco.MjModel.from_xml_path(model_path)
# data = mujoco.MjData(model)

loader = MJCFLoader(model_path)
# Make renderer, render and show the pixels
# renderer = mujoco.Renderer(model)
# mujoco.mj_forward(model, data)
# renderer.update_scene(data)

# media.show_image(renderer.render())

# Create a simulation instance
# sim = mujoco.MjSim(model)

# # Create a viewer to visualize the simulation
# viewer = mujoco.MjViewer(sim)

# Run the simulation for 1000 steps
# def step_fn(physics, random_state):
#     # Here you can apply forces, compute rewards, etc.
#     pass

# # Create a viewer and pass the physics and step function
# viewer.launch(model, step_fn)

# # Close the viewer
# viewer.close()

# loader = MJCFLoader(model_path)
# [print(k, v) for k, v in loader.default_class_dict.items()]












