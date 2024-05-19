
from simpub import SimPublisher, SimScene
from pathlib import Path

SERVICE_PORT = 5521 # this port is configured in the firewall 
DISCOVERY_PORT = 5520
STREAMING_PORT = 5522

import sys

scene = [scene for scene in Path("scenes").rglob("*/scene.xml")][int(sys.argv[1])]



# scene = "scenes/anybotics_anymal_c/scene.xml"
# scene = "scenes/anybotics_anymal_b/scene.xml"
# scene = "scenes/franka_emika_panda/scene.xml"
print("Loading scene", str(scene))

scene = SimScene.from_file(str(scene))

publisher = SimPublisher(scene)
publisher.start()

while True:
  ...
