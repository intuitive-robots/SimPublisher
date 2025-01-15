import random
from pprint import pprint
import time

from simpub.core import init_node
from simpub.core.net_component import Publisher, StrService


def random_string(name: str) -> str:
    return f"test_{name}_{random.randint(0, 1000)}"


if __name__ == "__main__":
    manager = init_node(random_string("node"), "192.168.0.134")
    time.sleep(1)
    pub_1 = Publisher(random_string("pub"), True)
    service_1 = StrService(random_string("service"), lambda x: x)
    pprint(manager.local_info)
    manager.spin()
