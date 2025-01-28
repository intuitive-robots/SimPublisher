import random

from simpub.core import init_node
from simpub.xr_device.meta_quest3 import MetaQuest3

manager = init_node("192.168.0.134", f"test_node_{random.randint(0, 1000)}")
manager.start_node_broadcast()
meta = MetaQuest3("IRLMQ3-1")
manager.spin()
