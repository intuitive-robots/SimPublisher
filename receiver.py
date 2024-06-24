from simpub.receiver import SimReceiver
from simpub.unity_data import UnityScene


recv = SimReceiver()

recv.start()


@recv.on("INIT")
def on_init(scene : UnityScene):
  print(scene)


@recv.on("UPDATE")
def on_update(msg):
  print("Update")