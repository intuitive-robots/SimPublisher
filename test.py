
from simpub import SimPublisher
from simpub.loaders.json import JsonScene
from simpub.mujoco import mjcf_parser


# scene = [scene for scene in Path("scenes").rglob("*/scene.xml")][int(sys.argv[1])]


# scene = "scenes/anybotics_anymal_b/scene.xml"
scene = "/home/jlz/fun/fancy_gym/fancy_gym/envs/mujoco/table_tennis/assets/xml/table_tennis_env.xml"

scene = "scenes/franka_emika_panda/scene.xml"
# scene = "/home/jlz/fun/fancy_gym/fancy_gym/envs/mujoco/box_pushing/assets/box_pushing.xml"
print("Loading scene", str(scene))

scene = mjcf_parser.MJCFScene.from_file(scene, set(range(3)))
print(scene)

with open("dump.json", "w") as fp:
  fp.write(JsonScene.to_string(scene))

publisher = SimPublisher(scene)

  
publisher.start()

while True:
  ...

# import mujoco_py
# import numpy as np

# # Load the model
# model = mujoco_py.load_model_from_path('path_to_your_model.xml')

# # Create the simulation

# sim = mujoco_py.MjSim(model)

# # Set up visualization
# viewer = mujoco_py.MjViewer(sim)

# # Run the simulation
# while True:
#     sim.step()
#     viewer.render()

