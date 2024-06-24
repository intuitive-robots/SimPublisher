
from simpub import SimPublisher
from simpub.mjcf.json import JsonScene
from simpub.mjcf import mjcf_parser


# scene = [scene for scene in Path("scenes").rglob("*/scene.xml")][int(sys.argv[1])]


scene = "scenes/anybotics_anymal_b/scene.xml"
# scene = "/home/jlz/fun/fancy_gym/fancy_gym/envs/mujoco/table_tennis/assets/xml/table_tennis_env.xml"

# scene = "scenes/franka_emika_panda/scene.xml"
# scene = "/home/jlz/fun/fancy_gym/fancy_gym/envs/mujoco/box_pushing/assets/box_pushing.xml"
print("Loading scene", str(scene))

scene = mjcf_parser.MJCFScene.from_file(scene)
print(scene)

with open("dump.json", "w") as fp:
  fp.write(JsonScene.to_string(scene))

publisher = SimPublisher(scene)

  
publisher.start()





while True:
  ...
