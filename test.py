
from simpub import SimPublisher
from simpub.loaders.json import JsonScene
from simpub.loaders.mjcf import MJCFScene


from simpub.simdata import SimMaterial
# scene = [scene for scene in Path("scenes").rglob("*/scene.xml")][int(sys.argv[1])]


scene = "scenes/anybotics_anymal_c/scene.xml"
# scene = "scenes/anybotics_anymal_b/scene.xml"
scene = "scenes/franka_emika_panda/scene.xml"
print("Loading scene", str(scene))

scene = MJCFScene.from_file(str(scene))

with open("dump.json", "w") as fp:
  fp.write(JsonScene.to_string(scene))

publisher = SimPublisher(scene)
publisher.start()



while True:
  ...
