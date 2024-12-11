from simpub.core import init_node
from simpub.core.net_component import Publisher

manager = init_node("127.0.0.1")
# pub_1 = Publisher("test_topic_1")
manager.spin()
