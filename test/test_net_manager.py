import random

from simpub.core import init_node
from simpub.core.net_component import Publisher

manager = init_node("192.168.0.134", f"test_node_{random.randint(0, 1000)}")
pub_1 = Publisher(f"test_topic_{random.randint(0, 1000)}")
manager.spin()
