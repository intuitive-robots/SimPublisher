from simpub.receiver import SimReceiver
from simpub.simdata import SimScene


recv = SimReceiver()

recv.start()


@recv.on("INIT")
def on_init(scene : SimScene):
  print(scene)


@recv.on("UPDATE")
def on_update(msg):
  print("Update")