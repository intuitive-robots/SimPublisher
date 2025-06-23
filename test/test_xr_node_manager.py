import random

from simpub.core import init_xr_node_manager
# from simpub.xr_device.meta_quest3 import MetaQuest3

manager = init_xr_node_manager("192.168.0.134")
manager.start_discover_node_loop()
# meta = MetaQuest3("IRLMQ3-1")
manager.spin()
