import mujoco_py

# Load the model from an XML file
model = mujoco_py.load_model_from_path("/home/xinkai/project/SimPublisher/example/mujoco/pendulum.xml")
sim = mujoco_py.MjSim(model)

# Create a viewer to visualize the simulation
viewer = mujoco_py.MjViewer(sim)

# Simulate for 100 steps
for _ in range(10000):
    sim.step()
    viewer.render()

print("Simulation done")
