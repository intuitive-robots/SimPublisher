
from simpub import SimPublisher
from simpub.loaders.json import JsonScene
from simpub.loaders import mjcf_parser


# scene = [scene for scene in Path("scenes").rglob("*/scene.xml")][int(sys.argv[1])]


# scene = "scenes/anybotics_anymal_c/scene.xml"
scene = "scenes/anybotics_anymal_b/scene.xml"

# scene = "scenes/franka_emika_panda/scene.xml"
# scene = "/home/jlz/fun/fancy_gym/fancy_gym/envs/mujoco/box_pushing/assets/box_pushing.xml"
print("Loading scene", str(scene))

scene = mjcf_parser.MJCFFile(scene).to_scene()

with open("dump.json", "w") as fp:
  fp.write(JsonScene.to_string(scene))

publisher = SimPublisher(scene)
publisher.start()



while True:
  ...
